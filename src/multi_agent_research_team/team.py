"""
L'équipe complète (Chapitre 2) : superviseur + Recherche + Rédacteur + Vérificateur.

Chaque agent est un nœud du grand graphe qui lit/écrit des onglets du
dossier partagé (`TeamState`).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_team.research_agent import run_research
from multi_agent_research_team.state import TeamState, Verdict

logger = logging.getLogger(__name__)

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
- Rédige entièrement dans la langue de la question : aucun mot d'une autre langue
  ni d'un autre alphabet, titres compris.

Format : un titre, une courte introduction, puis des sections en Markdown."""

VERIFIER_PROMPT = """Tu es le Vérificateur d'une équipe de recherche. Tu es sceptique.
On te donne un rapport et le texte complet des sources qu'il cite.

Pour CHAQUE affirmation du rapport, vérifie que la source citée à côté dit vraiment
la même chose (mêmes chiffres, mêmes dates, même sens). Une affirmation est non
appuyée si sa source ne la mentionne pas, la contredit, ou ne dit qu'une partie.

Approuve le rapport uniquement si TOUTES les affirmations sont appuyées. Sinon,
liste chaque affirmation non appuyée, recopiée telle qu'elle apparaît dans le rapport."""

# Utilisé quand le rapport ne cite aucune URL alors que des sources existent
# (souvent hors sujet, ex. une entreprise fictive). Seul le cas "aucun fait
# affirmé" est acceptable sans source ; tout fait non sourcé reste rejeté.
NO_CITATION_PROMPT = """Tu es le Vérificateur d'une équipe de recherche. Tu es sceptique.
Le rapport ci-dessous ne cite AUCUNE source.

Un rapport sans source est acceptable UNIQUEMENT s'il n'affirme aucun fait : il se
contente de dire que l'information n'a pas été trouvée ou que les notes ne suffisent pas.
Approuve-le dans ce cas seulement. S'il affirme le moindre fait (un nom, une date, un
chiffre, une explication), rejette-le et liste chaque affirmation, recopiée telle quelle."""

# Livré tel quel, sans LLM, quand la Recherche n'a ramené aucune source.
NO_SOURCES_REPORT = (
    "Aucune source n'a pu être trouvée pour cette question : je ne peux pas y "
    "répondre de façon sourcée."
)

# Nombre maximal de réécritures après un rejet du Vérificateur (budget de la
# boucle de correction, distinct du plafond de recherches de l'agent Recherche).
MAX_REVISIONS = 2

# Filet de sécurité ULTIME sur le grand graphe, pas une logique métier (ce
# sont MAX_REVISIONS et le plafond de recherches qui bornent les boucles
# normales). Le pire cas normal fait 15 étapes (superviseur compris) ; au-delà
# de 30, c'est forcément un bug de routage : mieux vaut lever une
# GraphRecursionError tout de suite que tourner jusqu'au défaut de LangGraph
# (10 007 étapes) en consommant du quota LLM à chaque tour.
RECURSION_LIMIT = 30

# Une URL s'arrête au premier espace, crochet, chevron ou guillemet double.
# Apostrophe, parenthèses et virgule sont AUTORISÉES : Wikipédia en met dans ses
# URL ("Ballon_d'or_2025", "Michael_Jackson_(baseball_player)", "Paris,_Texas").
# Les exclure coupait l'URL, que le niveau 1 prenait pour une URL inventée --
# bugs trouvés les 16 et 17 nov en lisant les brouillons : la boucle de
# correction effaçait une bonne réponse correctement citée. Ce qui est en
# trop en FIN d'URL est retiré par `_clean_url`.
_URL_RE = re.compile(r"https?://[^\s\[\]<>\"]+")


def make_research_node(research_agent: CompiledStateGraph):
    """
    Adaptateur entre l'agent Recherche du Chapitre 1 (qui parle en
    `messages`) et le dossier (qui parle en onglets). L'agent est réutilisé
    tel quel : sa boucle interne (tool calls, résultats bruts) reste
    enfermée dans `run_research`, seul le produit fini remonte au dossier.
    """

    def research(state: TeamState) -> dict[str, Any]:
        out = run_research(state["question"], research_agent)  # lit : question
        # écrit : notes, sources, searches (ce dernier pour la mesure : Ch.5)
        return {"notes": out["notes"], "sources": out["sources"], "searches": out["searches"]}

    return research


def no_sources(state: TeamState) -> dict[str, Any]:
    """
    Zéro source : rien ne peut être sourcé, donc la seule réponse honnête est
    connue d'avance. Décidé par le CODE, sans Rédacteur : aucun LLM n'écrit,
    donc rien ne peut être inventé (avant, ce cas finissait en "rien trouvé"
    rejeté faute d'URL, puis 2 réécritures inutiles -- mesure du 2 nov).
    """
    # approved=True : le message n'affirme aucun fait, il n'y a rien à vérifier.
    return {"report": NO_SOURCES_REPORT, "verdict": Verdict(approved=True)}


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
                    # Les notes peuvent elles-mêmes être fausses (vu en trace,
                    # Ch.4 : un fait attribué par la Recherche à une source qui
                    # ne le dit pas). Sans cette précision, le Rédacteur
                    # obéissait à "utilise les notes" et gardait la phrase.
                    "Le Vérificateur a lu le texte complet des sources : son avis prime sur "
                    "les notes, qui peuvent contenir des erreurs. Supprime chacune de ces "
                    "affirmations, même si elle figure dans les notes. Ne change pas le reste.\n"
                    # Vu en réel (Ch.6, question NBA) : le rapport final disait
                    # "l'affirmation initiale ayant été rejetée par le
                    # Vérificateur" -- le lecteur ne connaît pas notre cuisine.
                    "Le lecteur ne connaît ni le Vérificateur ni les notes : ne mentionne "
                    "jamais la correction ni le processus. Si un élément n'a plus rien de "
                    "sourcé à dire, retire-le entièrement."
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


def _clean_url(url: str) -> str:
    """Retire la ponctuation de la phrase collée en fin d'URL."""
    while True:
        # Point final, virgule entre deux URL, apostrophe de fermeture
        # ('https://a.com')...
        url = url.rstrip(".;:,'")
        # ... et ")" seulement si elle n'a pas de "(" dans l'URL :
        # "(voir https://a.com)" -> "https://a.com", mais
        # ".../Michael_Jackson_(baseball_player)" reste intact.
        if url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
            continue
        return url


def extract_cited_urls(report: str) -> set[str]:
    return {_clean_url(url) for url in _URL_RE.findall(report)}


def _scripts(text: str) -> set[str]:
    """Alphabets des lettres du texte, d'après leur nom Unicode.

    "ARABIC LETTER TEH" -> ARABIC, "LATIN SMALL LETTER E WITH ACUTE" -> LATIN.
    MODIFIER exclu : l'apostrophe typographique ʼ est une "lettre" pour Unicode.
    """
    found = set()
    for ch in text:
        if ch.isalpha():
            script = unicodedata.name(ch, "UNKNOWN").split()[0]
            if script not in {"LATIN", "MODIFIER"}:
                found.add(script)
    return found


def foreign_script_words(report: str, reference: str) -> list[str]:
    """Mots du rapport écrits dans un alphabet absent de la question et des notes.

    Vu en réel (Ch.6) : "La capitale de l'Australie et son تاريخ" -- "histoire"
    en arabe, produit par le Rédacteur lui-même (ni les notes ni les sources n'en
    contenaient) : glissement de langue d'un petit modèle. Comparer à la
    référence évite de punir un nom en arabe légitimement présent dans les notes.
    """
    allowed = _scripts(reference)
    words = (w.strip(".,;:!?()[]«»\"'") for w in report.split())
    return sorted({w for w in words if _scripts(w) - allowed})


def make_verifier_node(llm: BaseChatModel):
    """
    Vérification en deux niveaux :
      1. CODE (gratuit, certain S'IL EST CORRECT -- cf. bug de l'apostrophe) :
         chaque URL citée doit être dans les sources réellement récupérées.
         Sinon -> rejet immédiat, sans LLM.
      2. LLM (jugement) : la source dit-elle vraiment ce que le rapport
         affirme ? Réponse forcée au format `Verdict` (structured output).
         Sans aucune citation : le LLM vérifie seulement que le rapport
         n'affirme aucun fait (aveu "rien trouvé" accepté, fait nu rejeté).
    """
    # with_structured_output : Gemini est OBLIGÉ de répondre en remplissant
    # le formulaire Verdict (validé par Pydantic), pas en texte libre --
    # c'est ce qui permet au superviseur (du code) de lire la décision.
    judge = llm.with_structured_output(Verdict)

    def ask_judge(prompt: str, content: str) -> dict[str, Any]:
        try:
            verdict = judge.invoke([SystemMessage(prompt), HumanMessage(content)])
        except Exception as exc:
            # Panne du LLM (vu en trace LangSmith : 2 x 30 s puis 504 au 3e
            # passage, et 4 min de travail perdues dans le plantage). On livre
            # le rapport NON VÉRIFIÉ plutôt que de tout perdre -- statut
            # distinct de "rejeté" : rien n'a été lu, donc aucun reproche.
            logger.warning("verifier: vérification impossible: %r", exc)
            return {
                "verdict": Verdict(approved=False, unsupported_claims=[]),
                "verification_error": f"{type(exc).__name__}: {exc}",
            }
        return {"verdict": verdict}  # écrit : verdict

    def verifier(state: TeamState) -> dict[str, Any]:
        report, sources = state["report"], state["sources"]  # lit : report, sources
        cited = extract_cited_urls(report)
        known = {s["url"] for s in sources}

        # --- Niveau 1 : code ---
        # Hors du rôle du juge LLM ("est-ce sourcé ?") : un mot dans une autre
        # langue n'est pas une affirmation, il passait donc approuvé.
        reference = f"{state.get('question', '')}\n{state.get('notes', '')}"
        foreign = foreign_script_words(report, reference)
        if foreign:
            claims = [
                f"Mot dans une autre langue que celle de la question : « {word} » "
                "(réécris-le dans la langue de la question)"
                for word in foreign
            ]
            return {"verdict": Verdict(approved=False, unsupported_claims=claims)}
        invented = sorted(cited - known)
        if invented:
            claims = [f"URL citée absente des sources récupérées : {url}" for url in invented]
            return {"verdict": Verdict(approved=False, unsupported_claims=claims)}
        if not cited:
            # Avant : rejet d'office. Mais "je n'ai rien trouvé" est honnête
            # (fausse alarme mesurée le 2 nov). Le code ne sait pas distinguer
            # un aveu d'ignorance d'un fait inventé sans source : seul ce cas
            # douteux part au LLM, avec une consigne dédiée.
            return ask_judge(NO_CITATION_PROMPT, f"Rapport :\n{report}")

        # --- Niveau 2 : LLM ---
        # On ne lui montre que les sources CITÉES, en texte complet (pas les
        # extraits tronqués du Chapitre 1) : c'est tout ce dont il a besoin.
        cited_sources = "\n\n---\n\n".join(
            f"URL: {s['url']}\n{s['content']}" for s in sources if s["url"] in cited
        )
        return ask_judge(
            VERIFIER_PROMPT, f"Rapport :\n{report}\n\nSources citées :\n{cited_sources}"
        )

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
    if not state.get("sources") and "report" not in state:
        return "no_sources"  # décision certaine : rien à rédiger ni à vérifier
    if "report" not in state:
        return "writer"
    verdict = state.get("verdict")
    if verdict is None:  # jamais vérifié, ou verdict effacé par une réécriture
        return "verifier"
    if verdict.approved:
        return END
    # Panne de vérification : réécrire ne corrigerait rien -> on s'arrête et
    # on livre le rapport marqué "non vérifié".
    if state.get("verification_error"):
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
    graph.add_node("no_sources", no_sources)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route)
    # Topologie en étoile : chaque agent rend la main au superviseur.
    for agent in ("research", "writer", "verifier", "no_sources"):
        graph.add_edge(agent, "supervisor")
    return graph.compile().with_config(recursion_limit=RECURSION_LIMIT, run_name="research_team")


def main() -> None:
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
    # metadata : attachées à la trace LangSmith, filtrables dans l'interface
    # (ex. comparer les exécutions par modèle).
    config = {"metadata": {"model": llm.model, "question": question}}
    for step in team.stream({"question": question}, config, stream_mode="updates"):
        for node, update in step.items():
            print(f"--> {node}")
            final.update(update or {})

    print(f"\n=== Rapport ===\n{final['report']}")
    verdict = final["verdict"]
    if final.get("verification_error"):
        status = f"⚠️ NON VÉRIFIÉ -- vérification impossible ({final['verification_error'][:80]})"
    else:
        status = "APPROUVÉ" if verdict.approved else "REJETÉ"
    print(f"\n=== Verdict : {status} (après {final['revisions']} correction(s)) ===")
    for claim in verdict.unsupported_claims:
        print(f"- {claim}")


if __name__ == "__main__":
    main()
