from fakes import FakeSearch, fake_model, search_call, verdict_call
from langchain_core.messages import AIMessage

from multi_agent_research_team.research_agent import build_research_agent
from multi_agent_research_team.state import Verdict
from multi_agent_research_team.team import (
    extract_cited_urls,
    make_research_node,
    make_verifier_node,
    make_writer_node,
)
from multi_agent_research_team.tools import make_search_tool

SRC_A = {"url": "https://a.com", "title": "A", "content": "fait A"}


def test_le_noeud_recherche_ne_verse_au_dossier_que_notes_et_sources():
    llm = fake_model([search_call("q1", "c1"), AIMessage("- fait A [https://a.com]")])
    agent = build_research_agent(llm, [make_search_tool(FakeSearch({"q1": [SRC_A]}))])
    research = make_research_node(agent)

    update = research({"question": "question ?"})

    # Ni messages, ni tool calls : seulement le produit fini.
    assert update == {"notes": "- fait A [https://a.com]", "sources": [SRC_A]}


def test_le_redacteur_ne_voit_que_question_et_notes_et_ecrit_le_rapport():
    llm = fake_model([AIMessage("# Rapport\nFait A [https://a.com]")])
    writer = make_writer_node(llm)

    raw_source = {"url": "https://a.com", "title": "A", "content": "TEXTE BRUT DE LA PAGE"}

    update = writer(
        {"question": "question ?", "notes": "- fait A [https://a.com]", "sources": [raw_source]}
    )

    assert update == {"report": "# Rapport\nFait A [https://a.com]"}
    seen = llm.inputs[0][-1].content  # le message envoyé à Gemini
    assert "question ?" in seen
    assert "- fait A [https://a.com]" in seen
    assert "TEXTE BRUT DE LA PAGE" not in seen  # les sources brutes ne lui sont pas montrées


SOURCES = [
    {"url": "https://a.com", "title": "A", "content": "LangGraph a été créé en 2023."},
    {"url": "https://b.com", "title": "B", "content": "Texte de B, jamais cité."},
]


def test_extraction_des_urls_citees():
    report = "Fait 1 [https://a.com]. Fait 2 [https://a.com, https://b.com/page]."

    assert extract_cited_urls(report) == {"https://a.com", "https://b.com/page"}


def test_niveau1_une_url_inventee_est_rejetee_sans_appeler_le_llm():
    llm = fake_model([])  # aucune réponse scriptée : un appel ferait échouer le test
    verifier = make_verifier_node(llm)

    update = verifier({"report": "Fait [https://inventee.com]", "sources": SOURCES})

    assert update["verdict"].approved is False
    assert "https://inventee.com" in update["verdict"].unsupported_claims[0]
    assert llm.calls == []


def test_niveau1_un_rapport_sans_citation_est_rejete():
    verifier = make_verifier_node(fake_model([]))

    update = verifier({"report": "Des faits sans aucune source.", "sources": SOURCES})

    assert update["verdict"] == Verdict(approved=False, unsupported_claims=["Aucune source citée."])


def test_niveau2_le_llm_juge_avec_le_texte_complet_des_seules_sources_citees():
    claim = "LangGraph a été créé en 2024 [https://a.com]"
    llm = fake_model([verdict_call(False, [claim])])
    verifier = make_verifier_node(llm)

    update = verifier({"report": claim, "sources": SOURCES})

    assert update["verdict"] == Verdict(approved=False, unsupported_claims=[claim])
    seen = llm.inputs[0][-1].content
    assert "LangGraph a été créé en 2023." in seen  # texte complet de la source citée
    assert "Texte de B" not in seen  # source non citée : pas envoyée


def test_route_suit_les_onglets_remplis():
    from langgraph.graph import END

    from multi_agent_research_team.team import route

    assert route({"question": "q"}) == "research"
    assert route({"question": "q", "notes": "n", "sources": []}) == "writer"
    assert route({"question": "q", "notes": "n", "sources": [], "report": "r"}) == "verifier"
    verdict = Verdict(approved=True)
    assert (
        route({"question": "q", "notes": "n", "sources": [], "report": "r", "verdict": verdict})
        == END
    )


def test_equipe_complete_trajectoire_et_dossier_final():
    from multi_agent_research_team.team import build_team

    research_llm = fake_model([search_call("q1", "c1"), AIMessage("- fait A [https://a.com]")])
    research_agent = build_research_agent(
        research_llm, [make_search_tool(FakeSearch({"q1": [SRC_A]}))]
    )
    team_llm = fake_model([AIMessage("# Rapport\nFait A [https://a.com]"), verdict_call(True)])
    team = build_team(research_agent, team_llm)

    steps = list(team.stream({"question": "question ?"}, stream_mode="updates"))
    trajectory = [node for step in steps for node in step]

    assert trajectory == [
        "supervisor", "research",
        "supervisor", "writer",
        "supervisor", "verifier",
        "supervisor",
    ]  # fmt: skip

    final = {k: v for step in steps for update in step.values() for k, v in (update or {}).items()}
    assert final["notes"] == "- fait A [https://a.com]"
    assert final["sources"] == [SRC_A]
    assert final["report"] == "# Rapport\nFait A [https://a.com]"
    assert final["verdict"] == Verdict(approved=True)
