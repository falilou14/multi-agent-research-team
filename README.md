# Multi-Agent Research Team

Équipe multi-agents (superviseur + Recherche / Rédacteur / Vérificateur) qui répond à une question de recherche en interrogeant le web (Tavily), en rédigeant un rapport sourcé, puis en vérifiant que chaque affirmation est bien appuyée par une source citée — avec tracing (LangSmith) et une vraie suite d'évaluation (trajectoire + qualité du résultat), pas seulement "ça marche en démo".

Projet d'apprentissage (3ᵉ projet du portfolio), construit pour maîtriser *pourquoi* chaque choix technique a été fait, pas seulement le faire fonctionner — voir le guide de lecture avant de lire le code.

## Pourquoi ce projet

Un seul agent avec plusieurs rôles mélange des instructions contradictoires (être exhaustif en recherche vs concis en rédaction vs sceptique en vérification). Ici, chaque rôle est un agent séparé avec son propre prompt focalisé, orchestrés par un superviseur (LangGraph) qui décide qui appeler et dans quel ordre — y compris boucler en arrière si le Vérificateur rejette un rapport.

Le vrai sujet du projet n'est pas "faire marcher un agent" mais le rendre **observable** (LangSmith) et **mesurable** (suite d'évaluation sur un jeu de questions figé à l'avance) — les deux points qui manquent le plus souvent quand un système multi-agents passe de la démo à quelque chose de fiable.

Guide de lecture complet (topologies multi-agents, pourquoi un superviseur, pourquoi Tavily/LangSmith plutôt que des alternatives, méthodologie d'évaluation) : voir `guide-multi-agents-recherche` dans le dossier d'étude du portfolio.

## Architecture

- **Superviseur** — décide quel agent appeler ensuite (ou de terminer), à partir de l'état courant.
- **Agent Recherche** — interroge Tavily, renvoie des extraits sourcés (URL + passage).
- **Agent Rédacteur** — synthétise les extraits en rapport structuré avec citations.
- **Agent Vérificateur** — relit le rapport, détecte toute affirmation non supportée par une source citée ; peut déclencher un retour (recherche ou réécriture), plafonné.

## Stack

Python, LangGraph, Gemini (API, palier gratuit), Tavily (recherche web agent-native), LangSmith (observabilité), pytest, Ruff, GitHub Actions.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
pre-commit install
cp .env.example .env             # puis renseigner les clés
```

## Qualité & CI

```bash
ruff check .                                              # lint
ruff format .                                             # formatage
pytest -m "not integration" --cov=src                     # tests unitaires (sans vraie API)
pytest -m integration                                      # tests d'intégration réels -- à la main, jamais en CI (coût/quota)
```

La CI (GitHub Actions, `.github/workflows/ci.yml`) exécute lint + format check + tests unitaires sur chaque push/PR vers `main` — jamais d'appel à une vraie API externe en CI, pour éviter de griller le quota gratuit ou de dépendre d'une API tierce pour qu'une PR soit "verte".

## État

🚧 En cours — fondation (tests/lint/CI) posée, premier agent (Chapitre 1 : Recherche + Tavily) à venir.
