import pytest
from fakes import FakeSearch, fake_model, search_call
from langchain_core.messages import AIMessage

from multi_agent_research_team.research_agent import (
    build_prebuilt_research_agent,
    build_research_agent,
    run_research,
)
from multi_agent_research_team.tools import make_search_tool

SRC_A = {"url": "https://a.com", "title": "A", "content": "fait A"}
SRC_B = {"url": "https://b.com", "title": "B", "content": "fait B"}


def test_boucle_react_recherche_puis_conclut():
    client = FakeSearch({"q1": [SRC_A]})
    llm = fake_model([search_call("q1", "c1"), AIMessage("- fait A [https://a.com]")])
    agent = build_research_agent(llm, [make_search_tool(client)])

    out = run_research("question ?", agent)

    assert client.queries == ["q1"]
    assert out["searches"] == 1
    assert out["notes"] == "- fait A [https://a.com]"
    assert out["sources"] == [SRC_A]


def test_sans_tool_call_l_agent_termine_directement():
    client = FakeSearch()
    llm = fake_model([AIMessage("Rien à chercher.")])
    agent = build_research_agent(llm, [make_search_tool(client)])

    out = run_research("question ?", agent)

    assert client.queries == []
    assert out["searches"] == 0
    assert out["sources"] == []


def test_sources_dedoublonnees_par_url():
    client = FakeSearch({"q1": [SRC_A, SRC_B], "q2": [SRC_A]})
    llm = fake_model([search_call("q1", "c1"), search_call("q2", "c2"), AIMessage("fin")])
    agent = build_research_agent(llm, [make_search_tool(client)])

    out = run_research("question ?", agent)

    assert out["searches"] == 2
    assert [s["url"] for s in out["sources"]] == ["https://a.com", "https://b.com"]


def test_plafond_atteint_le_modele_est_rappele_sans_outils():
    client = FakeSearch()
    llm = fake_model([search_call("q1", "c1"), search_call("q2", "c2"), AIMessage("conclusion")])
    agent = build_research_agent(llm, [make_search_tool(client)], max_searches=2)

    out = run_research("question ?", agent)

    assert out["searches"] == 2
    # 2 appels avec outils liés, puis le 3e SANS -> forcé de conclure.
    assert llm.calls == [True, True, False]
    assert out["notes"] == "conclusion"


def test_le_prebuilt_parcourt_la_meme_boucle():
    client = FakeSearch({"q1": [SRC_A]})
    llm = fake_model([search_call("q1", "c1"), AIMessage("- fait A [https://a.com]")])
    agent = build_prebuilt_research_agent(llm, [make_search_tool(client)])

    out = run_research("question ?", agent)

    assert out["searches"] == 1
    assert out["sources"] == [SRC_A]


def test_le_prebuilt_refuse_la_recherche_de_trop_apres_coup():
    client = FakeSearch()
    llm = fake_model([search_call("q1", "c1"), search_call("q2", "c2"), AIMessage("conclusion")])
    agent = build_prebuilt_research_agent(llm, [make_search_tool(client)], max_searches=1)

    run_research("question ?", agent)

    # Le middleware bloque bien la 2e recherche (le moteur n'est appelé qu'une fois)...
    assert client.queries == ["q1"]
    # ...mais le modèle avait encore les outils au 2e appel : il a pu la demander.
    assert llm.calls == [True, True, True]


@pytest.mark.integration
def test_integration_vraie_recherche():
    from multi_agent_research_team.llm import get_llm

    agent = build_research_agent(get_llm(), [make_search_tool()])
    out = run_research("Quelle est la capitale de l'Australie ?", agent)

    assert 1 <= out["searches"] <= 3
    assert out["sources"]
    assert "Canberra" in out["notes"]
