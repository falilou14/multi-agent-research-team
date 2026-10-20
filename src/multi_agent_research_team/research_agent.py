"""
Agent Recherche (Chapitre 1) -- seul, avant toute complexité multi-agents.

Boucle ReAct construite À LA MAIN avec LangGraph :

    START -> agent --(tool_calls ?)--> tools -> agent -> ... -> END
                   |
                   +--(pas de tool_calls)-----------------------> END

`build_prebuilt_research_agent` construit la même chose avec `create_agent`
(langchain.agents -- remplace `create_react_agent` de langgraph.prebuilt,
déprécié depuis LangGraph 1.0) : ce que le prébuilt cache, c'est exactement
ce graphe-là, le plafond étant fourni par un middleware.
"""

from __future__ import annotations

import sys
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Plafond de recherches par question -- garde-fou de coût (quota/rate-limit du moteur) et
# de boucle infinie, indépendant du recursion_limit de LangGraph (qui, lui,
# lève une exception au lieu de laisser l'agent conclure proprement).
MAX_SEARCHES = 3

SYSTEM_PROMPT = f"""Tu es l'agent Recherche d'une équipe de recherche.
Ta seule mission : rassembler des informations factuelles et sourcées pour répondre
à la question de l'utilisateur. Tu ne rédiges pas de rapport final.

Méthode :
- Utilise l'outil `web_search` avec des requêtes courtes (mots-clés).
- Si les résultats sont insuffisants ou hors sujet, reformule et relance
  (au maximum {MAX_SEARCHES} recherches au total).
- Quand tu as assez d'éléments, arrête de chercher.

Réponse finale : une liste de faits clés, chacun suivi de l'URL de la source qui
l'appuie, au format `- fait [URL]`. N'invente jamais un fait ni une URL : tout ce que
tu écris doit venir des résultats de recherche. Si une information n'a pas été
trouvée, dis-le explicitement."""


def count_searches(messages: list) -> int:
    return sum(isinstance(m, ToolMessage) for m in messages)


def build_research_agent(
    llm: BaseChatModel, tools: list[BaseTool], max_searches: int = MAX_SEARCHES
) -> CompiledStateGraph:
    """
    Le LLM et les outils sont injectés (pas construits ici) : les tests
    passent un faux modèle et un faux outil, sans clé ni réseau.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict[str, Any]:
        # Plafond atteint -> on rappelle le modèle SANS outils : il ne peut
        # plus émettre de tool_call, il est forcé de conclure avec ce qu'il a.
        searches = count_searches(state["messages"])
        model = llm if searches >= max_searches else llm_with_tools
        response = model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        # On renvoie seulement le NOUVEAU message : le reducer `add_messages`
        # de MessagesState l'ajoute à la liste existante (pas d'écrasement).
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    # tools_condition : "tools" si le dernier AIMessage contient des
    # tool_calls, END sinon. C'est LA décision qui fait d'un graphe un agent.
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_prebuilt_research_agent(
    llm: BaseChatModel, tools: list[BaseTool], max_searches: int = MAX_SEARCHES
) -> CompiledStateGraph:
    """
    Même boucle via le prébuilt. Différence de plafond à noter :
    le middleware laisse le modèle DEMANDER une recherche de trop, puis la
    refuse (ToolMessage d'erreur) -- un appel LLM "gâché". Notre version
    retire les outils avant l'appel : le modèle ne peut même pas demander.
    """
    limiter = ToolCallLimitMiddleware(tool_name="web_search", run_limit=max_searches)
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT, middleware=[limiter])


def collect_sources(messages: list) -> list[dict[str, Any]]:
    """Sources structurées lues dans les artifacts des ToolMessages, dédoublonnées par URL."""
    seen: dict[str, dict[str, Any]] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.artifact:
            for source in m.artifact:
                seen.setdefault(source["url"], source)
    return list(seen.values())


def run_research(question: str, agent: CompiledStateGraph) -> dict[str, Any]:
    result = agent.invoke({"messages": [HumanMessage(question)]})
    messages = result["messages"]
    return {
        "notes": messages[-1].text,
        "sources": collect_sources(messages),
        "searches": count_searches(messages),
    }


def main() -> None:
    import logging

    from multi_agent_research_team.llm import get_llm
    from multi_agent_research_team.tools import make_search_tool

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # ddgs (via primp) et httpx logguent chaque requête HTTP : trop bavard.
    for noisy in ("primp", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    question = " ".join(sys.argv[1:]) or "Quelles sont les nouveautés de LangGraph 1.0 ?"
    agent = build_research_agent(get_llm(), [make_search_tool()])
    out = run_research(question, agent)

    print(f"\n=== Notes ({out['searches']} recherche(s)) ===\n{out['notes']}")
    print(f"\n=== Sources récupérées ({len(out['sources'])}) ===")
    for s in out["sources"]:
        print(f"- {s['title']} -- {s['url']}")


if __name__ == "__main__":
    main()
