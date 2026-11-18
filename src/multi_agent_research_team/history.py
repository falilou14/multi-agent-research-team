"""Historique des recherches (Ch.6), stocké dans SQLite.

Pourquoi SQLite : intégré à Python (rien à installer), un seul fichier, et
ACID -- une sauvegarde est entière ou absente, jamais à moitié, même si le
serveur plante au milieu. Limite : un seul serveur. Avec plusieurs serveurs
il faudrait PostgreSQL ; tout l'accès est isolé ici pour changer facilement.

Pas de comptes utilisateurs : chaque navigateur a un identifiant anonyme
(cookie), et CHAQUE requête filtre dessus. Sans ce filtre, changer un numéro
dans l'URL suffirait à lire l'historique d'un autre (faille IDOR).

Données personnelles (RGPD) : clear() = droit à l'effacement, et le fichier
de base ne va jamais dans git (data/ est ignoré).
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor    TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    status     TEXT    NOT NULL,
    report     TEXT    NOT NULL,
    sources    TEXT    NOT NULL,  -- JSON [{url, title}] : pour réafficher les citations
    revisions  INTEGER NOT NULL,
    duration_s REAL    NOT NULL,
    created_at TEXT    NOT NULL   -- ISO 8601, UTC
);
CREATE INDEX IF NOT EXISTS searches_by_visitor ON searches (visitor, id);
"""


class History:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # WAL : les lectures (liste de l'historique) ne bloquent pas
            # pendant qu'une autre requête enregistre une recherche.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # Une connexion par opération : FastAPI sert les requêtes dans
        # plusieurs threads, et une connexion sqlite3 ne se partage pas entre
        # threads. timeout : attend au lieu d'échouer si la base est occupée.
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def add(
        self,
        visitor: str,
        *,
        question: str,
        status: str,
        report: str,
        sources: list[dict[str, str]],
        revisions: int,
        duration_s: float,
    ) -> int:
        # `with conn` = une transaction : validée en entier, ou annulée en entier.
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO searches (visitor, question, status, report, sources,"
                " revisions, duration_s, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                # Paramètres "?" : jamais de texte collé dans le SQL (injection SQL).
                (
                    visitor,
                    question,
                    status,
                    report,
                    json.dumps(sources, ensure_ascii=False),
                    revisions,
                    round(duration_s, 1),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            return cursor.lastrowid

    def list(self, visitor: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, question, status, created_at FROM searches"
                " WHERE visitor = ? ORDER BY id DESC LIMIT ?",
                (visitor, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, visitor: str, search_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                # "AND visitor = ?" : LA ligne qui bloque l'IDOR.
                "SELECT * FROM searches WHERE id = ? AND visitor = ?",
                (search_id, visitor),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item.pop("visitor")  # inutile au client
        item["sources"] = json.loads(item["sources"])
        return item

    def delete(self, visitor: str, search_id: int) -> bool:
        """Supprime UNE recherche ; False si elle n'existe pas ou n'est pas à lui."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM searches WHERE id = ? AND visitor = ?",  # même garde que get()
                (search_id, visitor),
            )
            return cursor.rowcount == 1

    def clear(self, visitor: str) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM searches WHERE visitor = ?", (visitor,)).rowcount
