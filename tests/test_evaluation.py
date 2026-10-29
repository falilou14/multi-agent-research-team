from fakes import BrokenModel, FakeSearch, fake_model, judgement_call, search_call, verdict_call
from langchain_core.messages import AIMessage

from multi_agent_research_team.evaluation import (
    aggregate,
    is_quota_exhausted,
    load_cases,
    make_eval_judge,
    mentions_ok,
    run_case,
)
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


MJ_CASE = {
    "id": "premise-michael-jackson-basket",
    "family": "premisse_fausse",
    "question": "Michael Jackson est-il le meilleur basketteur ?",
    "expected_behavior": "Corrige la prémisse : chanteur, pas basketteur.",
    "must_mention": ["chanteur", "Jordan"],
}


def test_les_mots_attendus_ne_suffisent_pas_le_juge_tranche():
    # Piège proposé par Serigne Falilou : les bons mots, un sens faux.
    report = "Le chanteur Michael Jordan est le meilleur basketteur."
    llm = fake_model([judgement_call(False, "Fusionne les deux personnes.")])

    judgement = make_eval_judge(llm)(MJ_CASE, report)

    assert mentions_ok(report, MJ_CASE["must_mention"]) is True
    assert judgement.passed is False
    # Le juge voit le comportement attendu ET le rapport, jamais les sources.
    seen = llm.inputs[0][1].content
    assert MJ_CASE["expected_behavior"] in seen and report in seen


def test_juge_en_panne_donne_non_juge():
    judge = make_eval_judge(BrokenModel(TimeoutError("504")))

    assert judge(MJ_CASE, "rapport") is None


def test_run_case_enregistre_le_verdict_du_juge():
    team = make_team([AIMessage("Un seul Nobel, en 1921 [https://a.com]"), verdict_call(True)])
    judge = make_eval_judge(fake_model([judgement_call(True, "Prémisse corrigée.")]))

    result = run_case(team, CASE, judge)

    assert result["judge_passed"] is True
    assert result["judge_reason"] == "Prémisse corrigée."


def fiche(family, status="approuve", judge=True, mentions=True, crashed=False):
    if crashed:
        return {"family": family, "crashed": True}
    return {
        "family": family,
        "crashed": False,
        "status": status,
        "judge_passed": judge,
        "mentions_ok": mentions,
        "searches": 2,
        "revisions": 0,
        "duration_s": 10.0,
    }


def test_aggregate_calcule_les_taux_en_excluant_les_none():
    results = [
        fiche("introuvable", status="rejete", judge=True, mentions=None),
        fiche("introuvable", status="rejete", judge=None, mentions=None),
        fiche("recent"),
        fiche("recent", crashed=True),
    ]

    s = aggregate(results)

    assert s["total"]["crashed"] == (1, 4)
    assert s["total"]["approved"] == (1, 3)  # les plantages ne sont pas notés
    assert s["total"]["judge_passed"] == (2, 2)  # le "non jugé" est exclu
    assert s["total"]["mentions_ok"] == (1, 1)
    assert s["by_family"]["introuvable"]["cross"] == {
        "rejete / juge_ok": 1,
        "rejete / non_juge": 1,
    }


def test_le_jeu_reel_se_charge():
    cases = load_cases()

    assert len(cases) == 16
    assert any(c["id"] == "premise-michael-jackson-basket" for c in cases)


def test_quota_epuise_se_distingue_des_pannes_passageres():
    assert is_quota_exhausted("GoogleRateLimitError: 429 RESOURCE_EXHAUSTED. quota")
    assert not is_quota_exhausted("GoogleAPIError: 503 UNAVAILABLE. high demand")
