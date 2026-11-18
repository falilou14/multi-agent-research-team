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
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    # Bornes : une question vide ou un roman de 50 000 caractères = rejet 422
    # AVANT de dépenser le moindre appel Gemini.
    question: str = Field(min_length=3, max_length=500)


def to_event(node: str, update: dict[str, Any]) -> dict[str, Any] | None:
    """Traduit une étape interne en événement public (None = rien à montrer)."""
    if node == "research":
        return {
            "step": "research",
            "searches": update.get("searches", 0),
            # Lien + titre seulement : jamais le texte brut des pages.
            "sources": [
                {"url": s["url"], "title": s.get("title", "")} for s in update.get("sources", [])
            ],
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
    if state.get("verification_error"):
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


def create_app(team: Any) -> FastAPI:
    app = FastAPI(title="Équipe de recherche multi-agents")

    @app.get("/health")
    def health() -> dict[str, str]:
        # Ne touche PAS à l'équipe : répond même si Gemini est en panne.
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest) -> StreamingResponse:
        def events() -> Iterator[str]:
            state: dict[str, Any] = {"question": request.question}
            try:
                for step in team.stream(state.copy(), stream_mode="updates"):
                    for node, update in step.items():
                        state.update(update or {})
                        event = to_event(node, update or {})
                        if event:
                            yield sse(event)
                yield sse(final_event(state))
            except Exception as exc:
                # Le code HTTP 200 est déjà parti au 1er événement : une panne
                # en cours de route ne peut plus être un 500 -> événement "error".
                # Détail complet (pile, message) dans les journaux du SERVEUR ;
                # le client ne reçoit que le type : str(exc) peut contenir
                # projet, quota, chemins, voire une clé d'API dans une URL.
                logger.exception("Échec de /ask")
                yield sse({"step": "error", "message": type(exc).__name__})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
