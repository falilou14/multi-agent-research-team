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

    assert update == {"report": "# Rapport\nFait A [https://a.com]", "revisions": 0}
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


REJECTED = Verdict(approved=False, unsupported_claims=["affirmation fausse"])


def dossier(**tabs):
    return {"question": "q", **tabs}


def test_route_suit_les_onglets_remplis():
    from langgraph.graph import END

    from multi_agent_research_team.team import route

    assert route(dossier()) == "research"
    assert route(dossier(notes="n", sources=[])) == "writer"
    assert route(dossier(notes="n", sources=[], report="r")) == "verifier"
    approved = Verdict(approved=True)
    assert route(dossier(notes="n", sources=[], report="r", verdict=approved)) == END


def test_route_rejet_avec_budget_renvoie_au_redacteur():
    from multi_agent_research_team.team import route

    state = dossier(notes="n", sources=[], report="r", verdict=REJECTED, revisions=0)

    assert route(state) == "writer"


def test_route_verdict_efface_apres_reecriture_renvoie_au_verificateur():
    from multi_agent_research_team.team import route

    # Après réécriture, le Rédacteur a vidé l'onglet verdict (None) :
    # le NOUVEAU rapport doit être vérifié.
    state = dossier(notes="n", sources=[], report="r v2", verdict=None, revisions=1)

    assert route(state) == "verifier"


def test_route_rejet_budget_epuise_termine_quand_meme():
    from langgraph.graph import END

    from multi_agent_research_team.team import MAX_REVISIONS, route

    state = dossier(notes="n", sources=[], report="r", verdict=REJECTED, revisions=MAX_REVISIONS)

    assert route(state) == END


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


def test_premiere_redaction_initialise_le_compteur_sans_toucher_au_verdict():
    writer = make_writer_node(fake_model([AIMessage("rapport v1")]))

    update = writer(dossier(notes="- fait A [https://a.com]", sources=[]))

    assert update == {"report": "rapport v1", "revisions": 0}


def test_reecriture_recoit_les_reproches_efface_le_verdict_et_compte():
    llm = fake_model([AIMessage("rapport v2")])
    writer = make_writer_node(llm)

    update = writer(
        dossier(notes="- fait A [https://a.com]", sources=[], report="rapport v1",
                verdict=REJECTED, revisions=0)
    )  # fmt: skip

    assert update == {"report": "rapport v2", "verdict": None, "revisions": 1}
    seen = llm.inputs[0][-1].content
    assert "rapport v1" in seen  # son ancien rapport
    assert "affirmation fausse" in seen  # les reproches du Vérificateur


def run_team(team_responses):
    from multi_agent_research_team.team import build_team

    research_llm = fake_model([search_call("q1", "c1"), AIMessage("- fait A [https://a.com]")])
    research_agent = build_research_agent(
        research_llm, [make_search_tool(FakeSearch({"q1": [SRC_A]}))]
    )
    team = build_team(research_agent, fake_model(team_responses))
    steps = list(team.stream({"question": "question ?"}, stream_mode="updates"))
    trajectory = [node for step in steps for node in step if node != "supervisor"]
    final = {k: v for step in steps for update in step.values() for k, v in (update or {}).items()}
    return trajectory, final


def test_boucle_rejete_une_fois_puis_approuve():
    trajectory, final = run_team(
        [
            AIMessage("v1 [https://a.com]"), verdict_call(False, ["v1 fausse"]),
            AIMessage("v2 [https://a.com]"), verdict_call(True),
        ]
    )  # fmt: skip

    assert trajectory == ["research", "writer", "verifier", "writer", "verifier"]
    assert final["report"] == "v2 [https://a.com]"
    assert final["verdict"] == Verdict(approved=True)
    assert final["revisions"] == 1


def test_boucle_toujours_rejete_s_arrete_au_budget_avec_verdict_negatif():
    from multi_agent_research_team.team import MAX_REVISIONS

    responses = []
    for i in range(MAX_REVISIONS + 1):
        responses += [AIMessage(f"v{i + 1} [https://a.com]"), verdict_call(False, ["fausse"])]
    trajectory, final = run_team(responses)

    # 1 rédaction + MAX_REVISIONS réécritures, CHACUNE vérifiée.
    assert trajectory == ["research"] + ["writer", "verifier"] * (MAX_REVISIONS + 1)
    assert final["revisions"] == MAX_REVISIONS
    # Le rapport livré est le dernier, avec SON verdict (pas un verdict périmé).
    assert final["report"] == f"v{MAX_REVISIONS + 1} [https://a.com]"
    assert final["verdict"].approved is False


def test_filet_global_arrete_un_superviseur_bugge(monkeypatch):
    import itertools

    import pytest
    from langgraph.errors import GraphRecursionError

    from multi_agent_research_team import team as team_module

    # Bug simulé : un superviseur qui renvoie TOUJOURS au Rédacteur.
    monkeypatch.setattr(team_module, "route", lambda state: "writer")
    llm = fake_model(itertools.repeat(AIMessage("rapport")))  # réponses infinies
    team = team_module.build_team(build_research_agent(llm, []), llm)

    with pytest.raises(GraphRecursionError):
        team.invoke(dossier(notes="n", sources=[]))
    # Arrêté par le filet après ~RECURSION_LIMIT étapes, pas 10 007.
    assert len(llm.calls) <= team_module.RECURSION_LIMIT
