"""API HTTP de l'équipe de recherche (Ch.6).

L'API est la porte d'entrée du produit ; l'interface web n'en est qu'un client.
L'équipe est INJECTÉE (comme le LLM dans build_team) : les tests passent une
fausse équipe, sans Gemini ni DuckDuckGo.

Contrat d'API : on expose un format PUBLIC choisi (to_event), jamais l'état
interne (texte brut des pages, objets Pydantic) -- renommer un champ de
TeamState ne doit pas casser l'interface.
"""

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from multi_agent_research_team.history import History

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

# Cookie du visiteur anonyme : 32 caractères hexadécimaux (uuid4), rien d'autre.
VISITOR_COOKIE = "visitor"
_VISITOR_RE = re.compile(r"^[0-9a-f]{32}$")

# Budget par visiteur. Une question ~ 2 crédits Tavily + 4-6 appels Gemini ;
# palier gratuit Tavily ~ 1000 crédits/mois (~33/jour) : sans plafond, un seul
# script épuise le mois en quelques minutes et l'appli meurt pour tous.
MAX_QUESTIONS_PER_WINDOW = 10
WINDOW_S = 3600.0


class RateLimiter:
    """Fenêtre glissante par clé (IP) : au plus `max_requests` sur `window_s`.

    En mémoire : remis à zéro au redémarrage et propre à UN processus (plusieurs
    workers ou serveurs -> il faudrait un magasin partagé comme Redis).
    """

    def __init__(
        self,
        max_requests: int = MAX_QUESTIONS_PER_WINDOW,
        window_s: float = WINDOW_S,
        clock: Callable[[], float] = time.monotonic,  # injectable : tests sans attendre
    ):
        self.max_requests, self.window_s, self.clock = max_requests, window_s, clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        # Les routes synchrones de FastAPI tournent dans un pool de threads :
        # deux requêtes simultanées de la même IP ne doivent pas passer toutes
        # les deux sur la dernière place libre.
        self.lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """None = autorisé (et compté) ; sinon secondes à attendre."""
        with self.lock:
            now = self.clock()
            hits = self.hits[key]
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()  # plus vieux qu'une fenêtre : oublié
            if len(hits) >= self.max_requests:
                return hits[0] + self.window_s - now  # libéré quand le plus vieux expire
            hits.append(now)
            return None


class AskRequest(BaseModel):
    # Bornes : une question vide ou un roman de 50 000 caractères = rejet 422
    # AVANT de dépenser le moindre appel Gemini.
    question: str = Field(min_length=3, max_length=500)


def public_sources(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    # Lien + titre nettoyé (vu en réel : "…\r\n        \r\n – Parliament").
    return [{"url": s["url"], "title": " ".join(s.get("title", "").split())} for s in sources]


def to_event(node: str, update: dict[str, Any]) -> dict[str, Any] | None:
    """Traduit une étape interne en événement public (None = rien à montrer)."""
    if node == "research":
        return {
            "step": "research",
            "searches": update.get("searches", 0),
            # Lien + titre seulement : jamais le texte brut des pages. Titres
            # nettoyés (vu en réel : "…\r\n        \r\n – Parliament of Australia").
            "sources": public_sources(update.get("sources", [])),
        }
    if node == "writer":
        return {"step": "writer", "revisions": update.get("revisions", 0)}
    if node == "verifier":
        if update.get("verification_error"):
            return {"step": "verifier", "status": "unverified"}
        verdict = update["verdict"]
        return {
            "step": "verifier",
            "status": "approved" if verdict.approved else "rejected",
            "unsupported_claims": verdict.unsupported_claims,
        }
    if node == "no_sources":
        return {"step": "no_sources"}
    return None  # superviseur : aiguillage interne, rien pour l'utilisateur


def final_event(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("sources"):
        # En interne, no_sources pose approved=True (zéro affirmation = rien
        # de non sourcé). Pour un humain, une coche verte sur une non-réponse
        # serait trompeuse : le contrat traduit en statut à part (badge neutre).
        status = "no_sources"
    elif state.get("verification_error"):
        status = "unverified"
    else:
        status = "approved" if state["verdict"].approved else "rejected"
    return {
        "step": "done",
        "status": status,
        "report": state["report"],
        "revisions": state.get("revisions", 0),
    }


def sse(event: dict[str, Any]) -> str:
    # Format Server-Sent Events : "data: <json>" puis une ligne vide.
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def visitor_of(http: Request) -> str | None:
    """Identifiant du cookie s'il est bien formé ; un cookie trafiqué est ignoré."""
    value = http.cookies.get(VISITOR_COOKIE, "")
    return value if _VISITOR_RE.match(value) else None


def create_app(
    team: Any, limiter: RateLimiter | None = None, history: History | None = None
) -> FastAPI:
    """history=None : pas d'historique (tests de l'équipe, ou base désactivée)."""
    app = FastAPI(title="Équipe de recherche multi-agents")
    limiter = limiter or RateLimiter()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # L'interface n'est qu'un client de /ask, servie par la même API
        # (même origine : pas de CORS à configurer).
        return FileResponse(INDEX_HTML)

    @app.get("/health")
    def health() -> dict[str, str]:
        # Ne touche PAS à l'équipe : répond même si Gemini est en panne.
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest, http: Request) -> StreamingResponse:
        # AVANT de lancer l'équipe : un refus ne coûte rien. Ici le code HTTP
        # n'est pas encore parti, donc un vrai 429 est possible (pas un événement).
        # Limite connue : derrière un proxy, toutes les requêtes ont l'IP du proxy.
        ip = http.client.host if http.client else "inconnu"
        retry_after = limiter.check(ip)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Trop de questions : réessaie plus tard.",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )

        visitor = visitor_of(http) or uuid.uuid4().hex

        def events() -> Iterator[str]:
            state: dict[str, Any] = {"question": request.question}
            started = time.monotonic()
            try:
                for step in team.stream(state.copy(), stream_mode="updates"):
                    for node, update in step.items():
                        state.update(update or {})
                        event = to_event(node, update or {})
                        if event:
                            yield sse(event)
                done = final_event(state)
                # Enregistré AVANT d'envoyer "done" : si l'onglet se ferme juste
                # après, le générateur n'est plus tiré (expérience du 18 nov.).
                done["history_id"] = save(visitor, request.question, done, state, started)
                yield sse(done)
            except Exception as exc:
                # Le code HTTP 200 est déjà parti au 1er événement : une panne
                # en cours de route ne peut plus être un 500 -> événement "error".
                # Détail complet (pile, message) dans les journaux du SERVEUR ;
                # le client ne reçoit que le type : str(exc) peut contenir
                # projet, quota, chemins, voire une clé d'API dans une URL.
                logger.exception("Échec de /ask")
                yield sse({"step": "error", "message": type(exc).__name__})

        response = StreamingResponse(events(), media_type="text/event-stream")
        response.set_cookie(
            VISITOR_COOKIE,
            visitor,
            max_age=365 * 24 * 3600,
            httponly=True,  # illisible par JavaScript : un script injecté ne le vole pas
            samesite="lax",
            # secure=True à activer en production (HTTPS) ; ici localhost en HTTP.
        )
        return response

    def save(visitor: str, question: str, done: dict, state: dict, started: float) -> int | None:
        if history is None:
            return None
        try:
            return history.add(
                visitor,
                question=question,
                status=done["status"],
                report=done["report"],
                sources=public_sources(state.get("sources", [])),
                revisions=done["revisions"],
                duration_s=time.monotonic() - started,
            )
        except Exception:
            # L'historique est secondaire : une base en panne ne doit pas
            # faire perdre la réponse que l'utilisateur attend.
            logger.exception("Historique : enregistrement impossible")
            return None

    @app.get("/history")
    def list_history(http: Request) -> dict[str, list]:
        visitor = visitor_of(http)
        if history is None or visitor is None:
            return {"items": []}
        return {"items": history.list(visitor)}

    @app.get("/history/{search_id}")
    def get_history(search_id: int, http: Request) -> dict[str, Any]:
        visitor = visitor_of(http)
        item = history.get(visitor, search_id) if history and visitor else None
        if item is None:
            # 404 aussi pour la recherche d'un AUTRE visiteur : on ne révèle
            # même pas qu'elle existe.
            raise HTTPException(status_code=404, detail="Recherche introuvable.")
        return item

    @app.delete("/history", status_code=204)
    def clear_history(http: Request) -> Response:
        visitor = visitor_of(http)
        if history is not None and visitor is not None:
            history.clear(visitor)  # droit à l'effacement (RGPD)
        return Response(status_code=204)

    return app


def build_app() -> FastAPI:
    """Application réelle (Gemini + recherche web), pour uvicorn --factory."""
    from multi_agent_research_team.llm import get_llm  # charge aussi le .env
    from multi_agent_research_team.research_agent import build_research_agent
    from multi_agent_research_team.team import build_team
    from multi_agent_research_team.tools import make_search_tool

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    llm = get_llm()
    team = build_team(build_research_agent(llm, [make_search_tool()]), llm)
    return create_app(team, history=History(os.environ.get("HISTORY_DB", "data/history.db")))
