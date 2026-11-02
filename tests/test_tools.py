import pytest
from fakes import FakeDDGS, FakeSearch, FakeTavilyClient
from langchain_core.messages import ToolMessage

from multi_agent_research_team.tools import (
    MAX_CHARS_PER_RESULT,
    ddgs_backend,
    make_search_tool,
    tavily_backend,
)

SOURCE = {"url": "https://example.com/a", "title": "Titre A", "content": "x" * 5000}


def call(tool, query: str) -> ToolMessage:
    # Invoqué avec un ToolCall complet (comme le fait ToolNode) -> ToolMessage.
    return tool.invoke(
        {"name": "web_search", "args": {"query": query}, "id": "1", "type": "tool_call"}
    )


def test_le_llm_voit_l_url_avant_chaque_extrait_tronque_sans_numero():
    msg = call(make_search_tool(FakeSearch({"q": [SOURCE]})), "q")

    assert isinstance(msg, ToolMessage)
    assert msg.content.startswith("URL: https://example.com/a\nTitre: Titre A")
    assert "[1]" not in msg.content
    assert len(msg.content) < MAX_CHARS_PER_RESULT + 100


def test_l_artifact_garde_les_donnees_structurees_completes():
    msg = call(make_search_tool(FakeSearch({"q": [SOURCE]})), "q")

    assert msg.artifact == [SOURCE]  # non tronqué : réservé au code, pas au LLM


def test_aucun_resultat_donne_un_message_explicite():
    msg = call(make_search_tool(FakeSearch()), "rien")

    assert msg.content == "Aucun résultat pour cette requête."
    assert msg.artifact == []


def test_une_panne_du_moteur_est_renvoyee_au_modele_sans_planter():
    tool = make_search_tool(ddgs_backend(FakeDDGS(error=TimeoutError("lent")), wait_s=0))

    msg = call(tool, "q")

    assert msg.content.startswith("Erreur de recherche (TimeoutError)")
    assert msg.artifact == []


def test_ddgs_instable_est_reessaye_sans_que_l_agent_le_voie():
    raw = {"href": "https://a.com", "title": "A", "body": "x"}
    client = FakeDDGS([raw], error=RuntimeError("No results found."), failures=2)

    results = ddgs_backend(client, wait_s=0)("q")

    assert client.calls == 3  # 2 échecs puis succès, dans le même appel
    assert results == [{"url": "https://a.com", "title": "A", "content": "x"}]


def test_ddgs_abandonne_apres_trois_essais():
    client = FakeDDGS(error=RuntimeError("No results found."))

    with pytest.raises(RuntimeError):
        ddgs_backend(client, wait_s=0)("q")
    assert client.calls == 3


def test_un_resultat_sans_url_est_ecarte():
    raws = [{"href": "", "title": "vide", "body": "x"}, {"href": "https://a.com", "body": "y"}]

    results = ddgs_backend(FakeDDGS(raws))("q")

    assert [r["url"] for r in results] == ["https://a.com"]


def test_les_deux_backends_normalisent_au_meme_format():
    ddgs_raw = {"href": "https://example.com/a", "title": "Titre A", "body": "x"}
    tavily_raw = {"url": "https://example.com/a", "title": "Titre A", "content": "x", "score": 0.9}
    expected = [{"url": "https://example.com/a", "title": "Titre A", "content": "x"}]

    assert ddgs_backend(FakeDDGS([ddgs_raw]))("q") == expected
    assert tavily_backend(FakeTavilyClient([tavily_raw]))("q") == expected
