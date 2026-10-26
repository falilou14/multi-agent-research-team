"""
Construction du modèle de chat Gemini, partagé par tous les agents.

On passe par ChatGoogleGenerativeAI (langchain-google-genai) plutôt que par le
SDK google-genai brut : c'est ce qui donne accès à `bind_tools`, `ToolNode` et
`create_react_agent`, et ce que LangSmith sait tracer automatiquement (chaque
appel LLM, chaque tool call, les tokens) sans instrumentation manuelle.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY manquant -- copie .env.example vers .env et "
            "renseigne ta clé (gratuite sur https://aistudio.google.com/apikey)."
        )
    # Garde-fou réseau : par défaut (6 tentatives, backoff, pas de délai par
    # tentative), un 503 du palier gratuit peut bloquer un appel plusieurs
    # minutes. On borne explicitement.
    #
    # ATTENTION au nom trompeur : `max_retries` est transmis au SDK Google
    # comme un nombre TOTAL de tentatives (max_retries=1 = aucun nouvel
    # essai). Révélé par une trace LangSmith (Ch.4) : un seul appel de 20,2 s
    # avant "504 DEADLINE_EXCEEDED", soit notre propre timeout de 20 s.
    # Réglage retenu : 30 s par tentative, 2 tentatives -> une lenteur
    # passagère est absorbée, et ~1 min au pire avant une erreur claire.
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )
