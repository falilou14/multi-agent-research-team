"""
Harnais d'évaluation (Chapitre 5) -- le "banc d'essai" de l'équipe.

Il fait tourner la VRAIE équipe (vrai Gemini, vrai web) sur le jeu de
questions figé (evals/questions.json), plusieurs fois par question, et
relève pour chaque exécution :
  - la trajectoire (lue dans l'état final, par du code, gratuit) ;
  - la qualité de la réponse (mots attendus par du code, puis un juge LLM).

Tester != évaluer : les tests unitaires vérifient NOTRE code avec de faux
modèles ; le harnais mesure le COMPORTEMENT du système réel, en taux.
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph

DATASET = Path(__file__).resolve().parents[2] / "evals" / "questions.json"


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    # Minuscules + accents retirés : "Dembélé" doit matcher "dembele".
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def mentions_ok(report: str, must_mention: list[str]) -> bool | None:
    """
    Contrôle par le code : le rapport contient-il AU MOINS UN des mots
    attendus ? None quand la liste est vide (questions "introuvables") :
    il n'y a pas de bon mot à trouver, seul le juge LLM peut trancher.
    """
    if not must_mention:
        return None
    normalized = _normalize(report)
    return any(_normalize(word) in normalized for word in must_mention)


def final_status(state: dict[str, Any]) -> str:
    if state.get("verification_error"):
        return "non_verifie"
    verdict = state.get("verdict")
    if verdict is None:
        return "sans_verdict"  # ne devrait pas arriver : à investiguer si ça apparaît
    return "approuve" if verdict.approved else "rejete"


def run_case(team: CompiledStateGraph, case: dict[str, Any]) -> dict[str, Any]:
    """Une exécution de l'équipe sur un cas : jamais d'exception, tout est relevé."""
    start = time.monotonic()
    try:
        steps = list(team.stream({"question": case["question"]}, stream_mode="updates"))
    except Exception as exc:
        # Un plantage est un RÉSULTAT à compter (robustesse), pas une raison
        # d'arrêter toute l'évaluation au milieu.
        return {
            "id": case["id"],
            "family": case["family"],
            "crashed": True,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "duration_s": round(time.monotonic() - start, 1),
        }

    state: dict[str, Any] = {"question": case["question"]}
    trajectory = []
    for step in steps:
        for node, update in step.items():
            trajectory.append(node)
            state.update(update or {})

    report = state.get("report", "")
    return {
        "id": case["id"],
        "family": case["family"],
        "crashed": False,
        "duration_s": round(time.monotonic() - start, 1),
        "status": final_status(state),
        "revisions": state.get("revisions", 0),
        "searches": state.get("searches", 0),
        "trajectory": [n for n in trajectory if n != "supervisor"],
        "mentions_ok": mentions_ok(report, case["must_mention"]),
        "report": report,
        "unsupported_claims": state["verdict"].unsupported_claims if state.get("verdict") else [],
    }
