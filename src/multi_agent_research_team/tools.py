"""
Outil de recherche web exposé aux agents, avec backend interchangeable.

Séparation en deux couches :
  - un *backend* = une fonction `query -> list[Source]` qui parle à un moteur
    précis (DuckDuckGo, Tavily) et normalise sa réponse au format commun
    {url, title, content} ;
  - un *outil* unique `web_search` qui enveloppe n'importe quel backend.
L'agent ne connaît que l'outil : changer de moteur = changer SEARCH_BACKEND
dans .env, sans toucher au graphe ni aux prompts.

Écrit à la main plutôt que via `langchain-tavily` / `langchain-community` :
on contrôle exactement ce que le LLM voit (extraits tronqués, numérotés), ce
que le code récupère (artifact structuré), et on loggue chaque appel.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool

load_dotenv()

logger = logging.getLogger(__name__)

Source = dict[str, Any]  # {"url": str, "title": str, "content": str}
SearchFn = Callable[[str], list[Source]]

MAX_RESULTS = 5
# Un extrait peut faire plusieurs milliers de caractères (ddgs renvoie parfois
# un article Wikipedia entier) ; on tronque pour que 3 recherches x 5 résultats
# ne noient pas le contexte du LLM.
MAX_CHARS_PER_RESULT = 1200


DDGS_ATTEMPTS = 3
DDGS_RETRY_WAIT_S = 1.0


def ddgs_backend(client: Any = None, wait_s: float = DDGS_RETRY_WAIT_S) -> SearchFn:
    """DuckDuckGo via `ddgs` (métamoteur) : gratuit, sans clé, sans compte.
    Contrepartie : pas d'API officielle, donc rate-limit possible."""
    if client is None:
        from ddgs import DDGS

        client = DDGS(timeout=10)

    def search(query: str) -> list[Source]:
        # Mesure du 2 nov : "No results found" sur 18 recherches sur ~29,
        # alors que la même requête réussissait au 2e essai. On réessaie ICI,
        # invisible pour l'agent : sinon chaque panne lui coûte une recherche
        # de son budget de 3 ET un appel Gemini pour décider de relancer.
        for attempt in range(1, DDGS_ATTEMPTS + 1):
            try:
                results = client.text(query, max_results=MAX_RESULTS)
                break
            except Exception as exc:
                if attempt == DDGS_ATTEMPTS:
                    raise
                logger.info(
                    "ddgs: essai %d/%d échoué (%r), on réessaie", attempt, DDGS_ATTEMPTS, exc
                )
                time.sleep(wait_s * attempt)
        return [
            {"url": r["href"], "title": r.get("title", ""), "content": r.get("body", "")}
            for r in results
            # Vu en vrai : un résultat à l'URL vide. Sans URL, une source ne
            # peut être ni citée ni contrôlée par le Vérificateur (niveau 1).
            if r.get("href")
        ]

    return search


def tavily_backend(client: Any = None) -> SearchFn:
    """Tavily : API pensée pour les agents (extraits propres), clé requise."""
    if client is None:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY manquant (ou SEARCH_BACKEND=ddgs dans .env).")
        client = TavilyClient(api_key=api_key)

    def search(query: str) -> list[Source]:
        response = client.search(query, max_results=MAX_RESULTS, include_usage=True)
        # Suivi de quota (palier gratuit = crédits mensuels limités).
        logger.info("Tavily usage=%s", response.get("usage"))
        return [
            {"url": r["url"], "title": r.get("title", ""), "content": r.get("content", "")}
            for r in response.get("results", [])
        ]

    return search


BACKENDS: dict[str, Callable[[], SearchFn]] = {"ddgs": ddgs_backend, "tavily": tavily_backend}


def format_results(results: list[Source]) -> str:
    """
    Texte vu par le LLM. Volontairement SANS numéros ([1], [2]...) : au
    premier vrai test, le modèle citait "[3]" au lieu de l'URL -- or chaque
    recherche renumérote à partir de 1, donc "[3]" est ambigu dès la 2e
    recherche. Le modèle recopie ce que l'outil lui montre plus qu'il
    n'obéit au prompt : la seule chose citable doit être l'URL.
    """
    if not results:
        return "Aucun résultat pour cette requête."
    blocks = []
    for r in results:
        content = r["content"][:MAX_CHARS_PER_RESULT]
        blocks.append(f"URL: {r['url']}\nTitre: {r['title']}\n{content}")
    return "\n\n---\n\n".join(blocks)


def make_search_tool(search: SearchFn | None = None) -> BaseTool:
    """
    Le backend est injecté : les tests passent une fausse fonction de
    recherche (aucun appel réseau, aucune clé requise en CI). Sans argument,
    on prend celui désigné par SEARCH_BACKEND (ddgs par défaut).
    """
    if search is None:
        search = BACKENDS[os.environ.get("SEARCH_BACKEND", "ddgs")]()

    # response_format="content_and_artifact" : l'outil renvoie un tuple.
    #  - content  -> texte mis dans le ToolMessage, envoyé au LLM ;
    #  - artifact -> données structurées (url/titre/extrait) attachées au
    #    ToolMessage mais JAMAIS envoyées au LLM. C'est ce que le code
    #    récupère ensuite pour construire la liste des sources, sans avoir
    #    à re-parser le texte libre du modèle.
    @tool(response_format="content_and_artifact")
    def web_search(query: str) -> tuple[str, list[Source]]:
        """Recherche sur le web. Renvoie des extraits, chacun précédé de son URL.

        Args:
            query: requête de recherche courte et précise (mots-clés, pas une phrase entière).
        """
        try:
            results = search(query)
        except Exception as exc:
            # Panne réseau / rate-limit : on renvoie l'erreur AU MODÈLE plutôt
            # que de faire planter tout le graphe -- il peut reformuler ou
            # conclure avec ce qu'il a déjà (et le plafond borne les essais).
            logger.warning("web_search: query=%r a échoué: %r", query, exc)
            return f"Erreur de recherche ({type(exc).__name__}) : réessaie ou conclus.", []
        logger.info("web_search: query=%r results=%d", query, len(results))
        return format_results(results), results

    return web_search
