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
    # Même garde-fou réseau que dans le projet assurance : sur le palier
    # gratuit, un 503 (surcharge côté Google) peut sinon bloquer un appel
    # plusieurs minutes à cause des retries/backoff par défaut. On borne :
    # 20s par tentative, 1 seul retry -- l'agent échoue vite et proprement.
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        timeout=20,
        max_retries=1,
    )
