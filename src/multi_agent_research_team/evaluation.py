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
            "error": f"{type(exc).__name__}: {exc}"[:300],
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
