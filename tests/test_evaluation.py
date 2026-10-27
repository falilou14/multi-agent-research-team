from fakes import FakeSearch, fake_model, search_call, verdict_call
from langchain_core.messages import AIMessage

from multi_agent_research_team.evaluation import load_cases, mentions_ok, run_case
from multi_agent_research_team.research_agent import build_research_agent
from multi_agent_research_team.team import build_team
from multi_agent_research_team.tools import make_search_tool

SRC_A = {"url": "https://a.com", "title": "A", "content": "Einstein : Nobel 1921."}
CASE = {
    "id": "premise-einstein-second-nobel",
    "family": "premisse_fausse",
    "question": "Deuxième Nobel d'Einstein ?",
    "expected_behavior": "Corrige la prémisse.",
    "must_mention": ["1921"],
}


def make_team(team_responses):
    research_llm = fake_model([search_call("q1", "c1"), AIMessage("- Nobel 1921 [https://a.com]")])
    research_agent = build_research_agent(
        research_llm, [make_search_tool(FakeSearch({"q1": [SRC_A]}))]
    )
    return build_team(research_agent, fake_model(team_responses))


def test_mentions_ok_ignore_casse_et_accents():
    assert mentions_ok("Le lauréat est Ousmane DEMBELE.", ["Dembélé"]) is True
    assert mentions_ok("Aucun nom.", ["Dembélé", "Dembele"]) is False


def test_mentions_ok_sans_mot_attendu_laisse_le_juge_trancher():
    assert mentions_ok("N'importe quel rapport.", []) is None


def test_run_case_releve_la_trajectoire_et_la_qualite():
    team = make_team([AIMessage("Un seul Nobel, en 1921 [https://a.com]"), verdict_call(True)])

    result = run_case(team, CASE)

    assert result["crashed"] is False
    assert result["status"] == "approuve"
    assert result["searches"] == 1
    assert result["revisions"] == 0
    assert result["trajectory"] == ["research", "writer", "verifier"]
    assert result["mentions_ok"] is True


def test_run_case_un_plantage_est_un_resultat_pas_une_exception():
    team = make_team([])  # aucune réponse scriptée : le Rédacteur plante

    result = run_case(team, CASE)

    assert result["crashed"] is True
    assert result["error"]


def test_le_jeu_reel_se_charge():
    cases = load_cases()

    assert len(cases) == 16
    assert any(c["id"] == "premise-michael-jackson-basket" for c in cases)
