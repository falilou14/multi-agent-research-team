"""
Harnais d'évaluation (Chapitre 5) -- le "banc d'essai" de l'équipe.

Il fait tourner la VRAIE équipe (vrai Gemini, vrai web) sur le jeu de
questions figé (evals/questions.json), plusieurs fois par question, et
relève pour chaque exécution :
  - la trajectoire (lue dans l'état final, par du code, gratuit) ;
  - la qualité de la réponse (mots attendus par du code, puis un juge LLM).

Tester != évaluer : les tests unitaires vérifient NOTRE code avec de faux
modèles ; le harnais mesure le COMPORTEMENT du système réel, en taux.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATASET = Path(__file__).resolve().parents[2] / "evals" / "questions.json"

EVAL_JUDGE_PROMPT = """Tu es un évaluateur. On te donne une question, le comportement
attendu d'un bon rapport, et le rapport produit par un système de recherche.

Juge UNIQUEMENT si le rapport adopte le comportement attendu : bonne réponse,
prémisse fausse corrigée, refus d'inventer quand l'information n'existe pas.
Ne juge ni le style ni la longueur. Une réponse qui contient les bons mots mais
affirme quelque chose de faux échoue. Explique d'abord, puis tranche."""


class Judgement(BaseModel):
    # `reason` AVANT `passed` : on demande au juge de justifier puis de
    # trancher, plutôt que de décider d'abord et de rationaliser ensuite.
    reason: str = Field(description="Justification courte, en une ou deux phrases.")
    passed: bool = Field(description="True si le rapport adopte le comportement attendu.")


Judge = Callable[[dict[str, Any], str], Judgement | None]


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    # Minuscules + accents retirés : "Dembélé" doit matcher "dembele".
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def mentions_ok(report: str, must_mention: list[str]) -> bool | None:
    """
    Contrôle par le code : le rapport contient-il AU MOINS UN des mots
    attendus ? None quand la liste est vide (questions "introuvables") :
    il n'y a pas de bon mot à trouver, seul le juge LLM peut trancher.
    """
    if not must_mention:
        return None
    normalized = _normalize(report)
    return any(_normalize(word) in normalized for word in must_mention)


def final_status(state: dict[str, Any]) -> str:
    if state.get("verification_error"):
        return "non_verifie"
    verdict = state.get("verdict")
    if verdict is None:
        return "sans_verdict"  # ne devrait pas arriver : à investiguer si ça apparaît
    return "approuve" if verdict.approved else "rejete"


def make_eval_judge(llm: BaseChatModel) -> Judge:
    """
    Juge d'évaluation : "est-ce BIEN RÉPONDU ?" (≠ Vérificateur : "est-ce
    BIEN SOURCÉ ?"). Il ne voit pas les sources : il compare le rapport au
    comportement attendu, écrit par un humain qui connaît la vraie réponse.
    """
    judge = llm.with_structured_output(Judgement)

    def evaluate(case: dict[str, Any], report: str) -> Judgement | None:
        try:
            return judge.invoke(
                [
                    SystemMessage(EVAL_JUDGE_PROMPT),
                    HumanMessage(
                        f"Question : {case['question']}\n\n"
                        f"Comportement attendu : {case['expected_behavior']}\n\n"
                        f"Rapport :\n{report}"
                    ),
                ]
            )
        except Exception as exc:
            # Même logique que le Vérificateur en panne : "non jugé" est un
            # statut à part, jamais compté comme un échec du système.
            logger.warning("eval judge: jugement impossible: %r", exc)
            return None

    return evaluate


def run_case(
    team: CompiledStateGraph, case: dict[str, Any], judge: Judge | None = None
) -> dict[str, Any]:
    """Une exécution de l'équipe sur un cas : jamais d'exception, tout est relevé."""
    start = time.monotonic()
    try:
        steps = list(team.stream({"question": case["question"]}, stream_mode="updates"))
    except Exception as exc:
        # Un plantage est un RÉSULTAT à compter (robustesse), pas une raison
        # d'arrêter toute l'évaluation au milieu.
        return {
            "id": case["id"],
            "family": case["family"],
            "crashed": True,
            # 1000 caractères : à 300, le détail des erreurs de quota était coupé.
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "duration_s": round(time.monotonic() - start, 1),
        }

    state: dict[str, Any] = {"question": case["question"]}
    trajectory = []
    for step in steps:
        for node, update in step.items():
            trajectory.append(node)
            state.update(update or {})

    report = state.get("report", "")
    judgement = judge(case, report) if judge else None
    return {
        "id": case["id"],
        "family": case["family"],
        "crashed": False,
        "duration_s": round(time.monotonic() - start, 1),
        "status": final_status(state),
        "revisions": state.get("revisions", 0),
        "searches": state.get("searches", 0),
        "trajectory": [n for n in trajectory if n != "supervisor"],
        "mentions_ok": mentions_ok(report, case["must_mention"]),
        # None = pas de juge, ou juge en panne : "non jugé", exclu des taux.
        "judge_passed": judgement.passed if judgement else None,
        "judge_reason": judgement.reason if judgement else None,
        "report": report,
        "unsupported_claims": state["verdict"].unsupported_claims if state.get("verdict") else [],
    }


def is_quota_exhausted(error: str) -> bool:
    # 429 RESOURCE_EXHAUSTED : quota de l'API dépassé (≠ 503/504, pannes
    # passagères qui, elles, peuvent se résorber à la question suivante).
    return "RESOURCE_EXHAUSTED" in error


def _ratio(values: list[bool | None]) -> tuple[int, int]:
    # (réussites, total) en EXCLUANT les None ("non jugé", "pas de mot
    # attendu") : ils ne sont ni des succès ni des échecs.
    counted = [v for v in values if v is not None]
    return sum(counted), len(counted)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ran = [r for r in results if not r["crashed"]]
    mean = lambda key: round(sum(r[key] for r in ran) / len(ran), 1) if ran else None  # noqa: E731
    cross: dict[str, int] = {}
    for r in ran:
        judged = {True: "juge_ok", False: "juge_ko", None: "non_juge"}[r["judge_passed"]]
        cross[f"{r['status']} / {judged}"] = cross.get(f"{r['status']} / {judged}", 0) + 1
    return {
        "runs": len(results),
        "crashed": _ratio([r["crashed"] for r in results]),
        "approved": _ratio([r["status"] == "approuve" for r in ran]),
        "judge_passed": _ratio([r["judge_passed"] for r in ran]),
        "mentions_ok": _ratio([r["mentions_ok"] for r in ran]),
        "avg_searches": mean("searches"),
        "avg_revisions": mean("revisions"),
        "avg_duration_s": mean("duration_s"),
        # Vérificateur x juge : "rejete / juge_ok" = fausse alarme probable du
        # Vérificateur ; "approuve / juge_ko" = bien sourcé mais mal répondu.
        "cross": dict(sorted(cross.items())),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Les fiches -> des taux, au total et par famille de questions."""
    families = sorted({r["family"] for r in results})
    return {
        "total": _summarize(results),
        "by_family": {f: _summarize([r for r in results if r["family"] == f]) for f in families},
    }


def _fmt(pair: tuple[int, int]) -> str:
    k, n = pair
    return f"{k}/{n} ({k / n:.0%})" if n else "-"


def print_summary(summary: dict[str, Any]) -> None:
    rows = [("TOTAL", summary["total"]), *summary["by_family"].items()]
    print(
        f"\n{'famille':<18}{'plantages':>14}{'approuvés':>14}{'juge OK':>14}{'mots OK':>14}"
        f"{'rech.':>7}{'réécr.':>7}{'durée':>8}"
    )
    for name, s in rows:
        print(
            f"{name:<18}{_fmt(s['crashed']):>14}{_fmt(s['approved']):>14}"
            f"{_fmt(s['judge_passed']):>14}{_fmt(s['mentions_ok']):>14}"
            f"{s['avg_searches'] or 0:>7}{s['avg_revisions'] or 0:>7}{s['avg_duration_s'] or 0:>7}s"
        )
    print("\nVérificateur x juge :")
    for key, count in summary["total"]["cross"].items():
        print(f"  {key:<28}{count}")


def main() -> None:
    import argparse
    import os
    from datetime import datetime

    from multi_agent_research_team.llm import DEFAULT_MODEL, get_llm
    from multi_agent_research_team.research_agent import build_research_agent
    from multi_agent_research_team.team import build_team
    from multi_agent_research_team.tools import make_search_tool

    parser = argparse.ArgumentParser(description="Évalue l'équipe sur le jeu figé.")
    parser.add_argument("--runs", type=int, default=1, help="passages par question")
    parser.add_argument("--ids", help="ids séparés par des virgules (défaut : toutes)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cases = load_cases()
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    llm = get_llm()
    team = build_team(build_research_agent(llm, [make_search_tool()]), llm)
    # Modèle du juge configurable : un juge différent du modèle évalué limite
    # le biais "Gemini se note lui-même" (même modèle par défaut, faute de mieux).
    judge = make_eval_judge(get_llm(model=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_MODEL)))

    out = DATASET.parent / "results" / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.parent.mkdir(exist_ok=True)
    results = []
    total = len(cases) * args.runs
    for i, case in itertools.product(range(args.runs), cases):
        result = run_case(team, case, judge)
        if result["crashed"] and is_quota_exhausted(result["error"]):
            # Quota épuisé : ce n'est pas une mesure de NOTRE système, et
            # toutes les fiches suivantes échoueraient pareil en 2 s (vu le
            # 30 oct : 11 "plantages" sur 16 qui ne mesuraient que le quota).
            # On ne l'enregistre pas et on s'arrête.
            print(f"\nQuota Gemini épuisé à la question {case['id']} : arrêt.")
            break
        result["run"] = i
        results.append(result)
        # Écrit tout de suite : une interruption ne perd que la fiche en cours.
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        status = "PLANTÉ" if result["crashed"] else result["status"]
        judged = result.get("judge_passed")
        print(f"[{len(results)}/{total}] {case['id']}: {status}, juge={judged}")

    print_summary(aggregate(results))
    print(f"\nFiches détaillées : {out}")


if __name__ == "__main__":
    main()
