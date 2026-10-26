"""
L'équipe complète (Chapitre 2) : superviseur + Recherche + Rédacteur + Vérificateur.

Chaque agent est un nœud du grand graphe qui lit/écrit des onglets du
dossier partagé (`TeamState`).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_team.research_agent import run_research
from multi_agent_research_team.state import TeamState, Verdict

WRITER_PROMPT = """Tu es le Rédacteur d'une équipe de recherche.
À partir des notes de recherche fournies, rédige un rapport clair et structuré qui
répond à la question.

Règles strictes :
- Utilise UNIQUEMENT les faits présents dans les notes. N'ajoute rien de ta propre
  mémoire, même si tu es sûr que c'est vrai : un fait sans source est un défaut.
- Termine chaque affirmation par l'URL qui l'appuie, entre crochets : [URL].
  Ne cite que des URLs présentes dans les notes, jamais une autre.
- Si les notes ne suffisent pas à répondre à une partie de la question, écris-le
  explicitement au lieu de combler le vide.

Format : un titre, une courte introduction, puis des sections en Markdown."""

VERIFIER_PROMPT = """Tu es le Vérificateur d'une équipe de recherche. Tu es sceptique.
On te donne un rapport et le texte complet des sources qu'il cite.

Pour CHAQUE affirmation du rapport, vérifie que la source citée à côté dit vraiment
la même chose (mêmes chiffres, mêmes dates, même sens). Une affirmation est non
appuyée si sa source ne la mentionne pas, la contredit, ou ne dit qu'une partie.

Approuve le rapport uniquement si TOUTES les affirmations sont appuyées. Sinon,
liste chaque affirmation non appuyée, recopiée telle qu'elle apparaît dans le rapport."""

# Nombre maximal de réécritures après un rejet du Vérificateur (budget de la
# boucle de correction, distinct du plafond de recherches de l'agent Recherche).
MAX_REVISIONS = 2

# Une URL s'arrête au premier espace, crochet, parenthèse, virgule ou guillemet.
_URL_RE = re.compile(r"https?://[^\s\[\]()<>,\"']+")


def make_research_node(research_agent: CompiledStateGraph):
    """
    Adaptateur entre l'agent Recherche du Chapitre 1 (qui parle en
    `messages`) et le dossier (qui parle en onglets). L'agent est réutilisé
    tel quel : sa boucle interne (tool calls, résultats bruts) reste
    enfermée dans `run_research`, seul le produit fini remonte au dossier.
    """

    def research(state: TeamState) -> dict[str, Any]:
        out = run_research(state["question"], research_agent)  # lit : question
        return {"notes": out["notes"], "sources": out["sources"]}  # écrit : notes, sources

    return research


def make_writer_node(llm: BaseChatModel):
    """
    Pas d'outil, pas de boucle : un seul appel LLM. Le Rédacteur ne voit QUE
    la question et les notes -- ni les sources brutes (réservées au
    Vérificateur), ni le carnet interne de la Recherche (jeté).
    """

    def writer(state: TeamState) -> dict[str, Any]:
        task = f"Question : {state['question']}\n\nNotes :\n{state['notes']}"
        verdict = state.get("verdict")

        if verdict is None:  # première rédaction
            response = llm.invoke([SystemMessage(WRITER_PROMPT), HumanMessage(task)])
            return {"report": response.text, "revisions": 0}  # écrit : report, revisions

        # Réécriture après un rejet : il reçoit son ancien rapport + les
        # reproches précis du Vérificateur, pour corriger uniquement ça.
        complaints = "\n".join(f"- {claim}" for claim in verdict.unsupported_claims)
        response = llm.invoke(
            [
                SystemMessage(WRITER_PROMPT),
                HumanMessage(
                    f"{task}\n\nTon rapport précédent :\n{state['report']}\n\n"
                    f"Le Vérificateur l'a rejeté. Affirmations non appuyées :\n{complaints}\n\n"
                    "Réécris le rapport : corrige chacune de ces affirmations pour qu'elle "
                    "colle exactement à ses sources, ou supprime-la. Ne change pas le reste."
                ),
            ]
        )
        return {
            "report": response.text,
            # Le verdict portait sur l'ANCIEN rapport : on l'efface pour que
            # le superviseur renvoie le nouveau au Vérificateur.
            "verdict": None,
            "revisions": state.get("revisions", 0) + 1,
        }

    return writer


def extract_cited_urls(report: str) -> set[str]:
    # rstrip : une URL en fin de phrase embarque souvent le point final.
    return {url.rstrip(".;:") for url in _URL_RE.findall(report)}


def make_verifier_node(llm: BaseChatModel):
    """
    Vérification en deux niveaux :
      1. CODE (gratuit, infaillible) : chaque URL citée doit être dans les
         sources réellement récupérées. Sinon -> rejet immédiat, sans LLM.
      2. LLM (jugement) : la source dit-elle vraiment ce que le rapport
         affirme ? Réponse forcée au format `Verdict` (structured output).
    """
    # with_structured_output : Gemini est OBLIGÉ de répondre en remplissant
    # le formulaire Verdict (validé par Pydantic), pas en texte libre --
    # c'est ce qui permet au superviseur (du code) de lire la décision.
    judge = llm.with_structured_output(Verdict)

    def verifier(state: TeamState) -> dict[str, Any]:
        report, sources = state["report"], state["sources"]  # lit : report, sources
        cited = extract_cited_urls(report)
        known = {s["url"] for s in sources}

        # --- Niveau 1 : code ---
        if not cited:
            return {"verdict": Verdict(approved=False, unsupported_claims=["Aucune source citée."])}
        invented = sorted(cited - known)
        if invented:
            claims = [f"URL citée absente des sources récupérées : {url}" for url in invented]
            return {"verdict": Verdict(approved=False, unsupported_claims=claims)}

        # --- Niveau 2 : LLM ---
        # On ne lui montre que les sources CITÉES, en texte complet (pas les
        # extraits tronqués du Chapitre 1) : c'est tout ce dont il a besoin.
        cited_sources = "\n\n---\n\n".join(
            f"URL: {s['url']}\n{s['content']}" for s in sources if s["url"] in cited
        )
        verdict = judge.invoke(
            [
                SystemMessage(VERIFIER_PROMPT),
                HumanMessage(f"Rapport :\n{report}\n\nSources citées :\n{cited_sources}"),
            ]
        )
        return {"verdict": verdict}  # écrit : verdict

    return verifier


def route(state: TeamState) -> str:
    """
    Le superviseur (option A : des règles en code, pas un LLM). L'ordre est
    connu d'avance, donc un `if` décide gratuitement et sans erreur ; un
    superviseur LLM paierait un appel par étape pour la même décision.
    Il regarde quels onglets du dossier sont déjà remplis, puis le verdict.
    """
    if "notes" not in state:
        return "research"
    if "report" not in state:
        return "writer"
    verdict = state.get("verdict")
    if verdict is None:  # jamais vérifié, ou verdict effacé par une réécriture
        return "verifier"
    if verdict.approved:
        return END
    # Rejeté : on renvoie au Rédacteur tant qu'il reste du budget. Sans ce
    # plafond, un Vérificateur jamais satisfait ferait boucler à l'infini.
    if state.get("revisions", 0) < MAX_REVISIONS:
        return "writer"
    # Budget épuisé : on livre le rapport AVEC son verdict négatif, plutôt
    # que de le cacher -- l'utilisateur sait qu'il n'est pas fiable.
    return END


def build_team(research_agent: CompiledStateGraph, llm: BaseChatModel) -> CompiledStateGraph:
    graph = StateGraph(TeamState)
    # Le superviseur est un nœud qui ne modifie rien ({}) : toute sa
    # logique est dans `route`. En faire un vrai nœud (plutôt que de router
    # directement d'agent à agent) le rend visible dans la trajectoire et
    # dans les traces LangSmith, et lui donne une place pour tenir des
    # compteurs au Chapitre 3 (budget de la boucle de correction).
    graph.add_node("supervisor", lambda state: {})
    graph.add_node("research", make_research_node(research_agent))
    graph.add_node("writer", make_writer_node(llm))
    graph.add_node("verifier", make_verifier_node(llm))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route)
    # Topologie en étoile : chaque agent rend la main au superviseur.
    for agent in ("research", "writer", "verifier"):
        graph.add_edge(agent, "supervisor")
    return graph.compile()


def main() -> None:
    import logging
    import sys

    from multi_agent_research_team.llm import get_llm
    from multi_agent_research_team.research_agent import build_research_agent
    from multi_agent_research_team.tools import make_search_tool

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("primp", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    question = (
        " ".join(sys.argv[1:]) or "Quels sont les avantages et limites du pattern superviseur ?"
    )
    llm = get_llm()
    team = build_team(build_research_agent(llm, [make_search_tool()]), llm)

    # stream_mode="updates" : un événement par nœud exécuté -> la trajectoire.
    final: dict[str, Any] = {"question": question}
    for step in team.stream({"question": question}, stream_mode="updates"):
        for node, update in step.items():
            print(f"--> {node}")
            final.update(update or {})

    print(f"\n=== Rapport ===\n{final['report']}")
    verdict = final["verdict"]
    status = "APPROUVÉ" if verdict.approved else "REJETÉ"
    print(f"\n=== Verdict : {status} (après {final['revisions']} correction(s)) ===")
    for claim in verdict.unsupported_claims:
        print(f"- {claim}")


if __name__ == "__main__":
    main()
