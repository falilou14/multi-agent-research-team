"""
État partagé de l'équipe (Chapitre 2) -- le "dossier à onglets".

Contrairement au Chapitre 1 (un seul carnet `messages` où tout s'empile),
chaque onglet a UN agent qui l'écrit, et chaque agent ne lit que les onglets
dont il a besoin. La boucle interne de l'agent Recherche (tool calls,
résultats bruts) ne remonte jamais ici : seul son produit fini (notes +
sources) est versé au dossier.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """Avis du Vérificateur sur le rapport."""

    approved: bool = Field(description="True si chaque affirmation est appuyée par une source.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Affirmations du rapport non appuyées par les sources (vide si approuvé).",
    )


class TeamState(TypedDict):
    question: str  # écrit par : l'utilisateur, au départ
    # NotRequired : ces onglets sont VIDES au départ et remplis au fil du
    # travail. C'est justement leur présence/absence que lit le superviseur.
    notes: NotRequired[str]  # écrit par : Recherche
    sources: NotRequired[list[dict[str, Any]]]  # écrit par : Recherche (artifacts)
    report: NotRequired[str]  # écrit par : Rédacteur
    # écrit par : Vérificateur ; remis à None par le Rédacteur quand il
    # réécrit -- un verdict sur l'ANCIEN rapport est périmé et doit
    # disparaître, sinon le nouveau rapport ne serait jamais vérifié.
    verdict: NotRequired[Verdict | None]
    revisions: NotRequired[int]  # écrit par : Rédacteur (nombre de réécritures faites)
    # écrit par : Vérificateur, seulement si le LLM-juge est en panne. Distinct
    # d'un rejet : le rapport n'a pas été lu, il est livré "non vérifié".
    verification_error: NotRequired[str]
