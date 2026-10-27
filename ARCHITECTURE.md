# Architecture — référence rapide

## Le circuit

```
                      ┌──────────── rejeté + budget restant ───────────┐
                      ▼                                                 │
START → supervisor → research → writer → verifier ─┬─ approuvé ──────► FIN
                                                    ├─ rejeté, budget épuisé ──► FIN (verdict négatif)
                                                    ├─ panne du LLM-juge ──────► FIN (« non vérifié »)
                                                    └──────────────────────────────┘
```

Topologie en étoile : chaque agent rend la main au `supervisor`, qui décide de la suite (`route()` dans `team.py`).

## Qui utilise un LLM ?

| Composant | LLM ? | Ce qui travaille vraiment | Fichier |
|---|---|---|---|
| **Superviseur** | ❌ Non | Des `if` en Python (`route`) : l'ordre est connu d'avance | `team.py` |
| **Recherche** | ✅ Jusqu'à 4 appels | Gemini décide quoi chercher et écrit les notes | `research_agent.py` |
| ↳ outil `web_search` | ❌ Non | Du code qui interroge DuckDuckGo (ou Tavily) | `tools.py` |
| **Rédacteur** | ✅ 1 appel (+1 par réécriture) | Gemini écrit le rapport à partir des notes | `team.py` |
| **Vérificateur, niveau 1** | ❌ Non | Du code : chaque URL citée est-elle dans les sources récupérées ? | `team.py` |
| **Vérificateur, niveau 2** | ✅ 1 appel | Gemini juge si chaque source dit vraiment ce que le rapport affirme | `team.py` |

Règle suivie partout : **ce qu'un `if` peut décider, on ne le confie pas à un LLM**. Le LLM ne sert que là où il faut du jugement.

## Un seul Gemini, trois rôles

```python
llm = get_llm()  # UN seul Gemini
team = build_team(build_research_agent(llm, [...]), llm)  # partagé par les trois agents
```

Les trois agents utilisent **le même modèle**. Ce qui les différencie : leurs **consignes** (`SYSTEM_PROMPT`, `WRITER_PROMPT`, `VERIFIER_PROMPT`) et **ce qu'on leur donne à lire** (onglets du dossier partagé).

- **Conséquence** : ses défauts sont partagés. Si Gemini « sait » un fait par mémoire, la Recherche peut l'écrire en l'attribuant à une source qui ne le dit pas. C'est pourquoi le Vérificateur juge **par rapport aux sources récupérées**, pas par rapport à ce qu'il « sait ».
- **Évolution possible** : grâce à l'injection, un modèle différent par agent (ex. plus puissant pour le Vérificateur, plus rapide pour la Recherche) se fait sans toucher aux agents.

## Le dossier partagé (`TeamState`)

| Onglet | Écrit par | Lu par |
|---|---|---|
| `question` | l'utilisateur | tout le monde |
| `notes` | Recherche | Rédacteur |
| `sources` (artifacts, texte complet) | Recherche | Vérificateur |
| `report` | Rédacteur | Vérificateur |
| `verdict` | Vérificateur (remis à `None` par le Rédacteur à chaque réécriture) | Superviseur |
| `revisions` | Rédacteur | Superviseur |
| `verification_error` | Vérificateur (seulement en cas de panne) | Superviseur |

Chaque agent ne reçoit que le minimum dont il a besoin. Le carnet interne de la Recherche ne remonte jamais au dossier (il reste visible dans LangSmith).

## Les garde-fous

| Garde-fou | Protège contre | Où |
|---|---|---|
| Retrait des outils au plafond (3 recherches) | Demandes de recherche inutiles | case `agent`, `research_agent.py` |
| Exécution dans la limite du budget | Dépassement par appels parallèles | case `tools`, `research_agent.py` |
| `MAX_REVISIONS = 2` | Boucle Rédacteur ⇄ Vérificateur infinie | `route()`, `team.py` |
| Vérificateur niveau 1 avant le LLM | URL inventée validée par erreur | `team.py` |
| Panne du juge → « non vérifié » | Travail perdu sur un plantage | `team.py` |
| `timeout=30, max_retries=2` (= 2 tentatives au total) | Attente réseau (503/504 Gemini) | `llm.py` |
| `RECURSION_LIMIT = 30` | Bug de routage imprévu | `build_team`, `team.py` |

## Observabilité

LangSmith trace chaque exécution (`LANGCHAIN_TRACING_V2=true` dans `.env`) : graphes nommés `research_team` / `research_agent`, métadonnées `model` et `question`. Les tests désactivent le tracing (`tests/conftest.py`).
