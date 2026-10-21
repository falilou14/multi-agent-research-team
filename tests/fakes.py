"""Doublures de test : faux modèle de chat et faux moteurs de recherche, sans réseau ni clé."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolModel(GenericFakeChatModel):
    """
    GenericFakeChatModel rejoue une liste de réponses scriptées mais ne
    supporte pas bind_tools. On l'ajoute, et on enregistre à chaque appel si
    les outils étaient liés -- pour vérifier le plafond de recherches.
    """

    tools_bound: bool = False
    calls: list[bool] = []
    # Messages reçus à chaque appel -- pour vérifier ce qu'un agent "voit".
    inputs: list[list] = []

    def bind_tools(self, tools, **kwargs):
        # Copie superficielle : l'itérateur de réponses et les listes `calls`
        # / `inputs` sont partagés entre le modèle "nu" et "avec outils".
        return self.model_copy(update={"tools_bound": True})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(self.tools_bound)
        self.inputs.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def fake_model(responses: list[AIMessage]) -> FakeToolModel:
    return FakeToolModel(messages=iter(responses), calls=[], inputs=[])


def search_call(query: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "web_search", "args": {"query": query}, "id": call_id}]
    )


class FakeSearch:
    """Backend de recherche déjà normalisé (url/title/content) : enregistre les requêtes."""

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]] | None = None):
        self.results_by_query = results_by_query or {}
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.results_by_query.get(query, [])


class FakeTavilyClient:
    def __init__(self, results: list[dict[str, Any]]):
        self.results = results

    def search(self, query: str, **kwargs) -> dict[str, Any]:
        return {"results": self.results, "usage": {"credits": 1}}


class FakeDDGS:
    def __init__(self, results: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.results = results or []
        self.error = error

    def text(self, query: str, **kwargs) -> list[dict[str, Any]]:
        if self.error:
            raise self.error
        return self.results


def verdict_call(approved: bool, unsupported_claims: list[str] | None = None) -> AIMessage:
    """Réponse de Gemini qui "remplit le formulaire" Verdict (structured output via tool call)."""
    args = {"approved": approved, "unsupported_claims": unsupported_claims or []}
    return AIMessage(content="", tool_calls=[{"name": "Verdict", "args": args, "id": "v1"}])
