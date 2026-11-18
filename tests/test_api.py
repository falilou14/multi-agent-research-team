import json

from fastapi.testclient import TestClient

from multi_agent_research_team.api import create_app
from multi_agent_research_team.state import Verdict

SRC = {"url": "https://a.com", "title": "A", "content": "TEXTE BRUT DE LA PAGE"}


class FakeTeam:
    """Rejoue une suite d'étapes comme team.stream(stream_mode="updates")."""

    def __init__(self, steps, error=None):
        self.steps, self.error, self.calls = steps, error, 0

    def stream(self, inputs, config=None, stream_mode=None):
        self.calls += 1
        yield from self.steps
        if self.error:
            raise self.error


def events_of(response):
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]


HAPPY = [
    {"research": {"notes": "- A", "sources": [SRC], "searches": 1}},
    {"supervisor": None},
    {"writer": {"report": "Rapport [https://a.com]", "revisions": 0}},
    {"verifier": {"verdict": Verdict(approved=True)}},
]


def test_health_repond_sans_toucher_a_l_equipe():
    # object() : une "équipe" qui planterait au moindre appel -> prouve que
    # /health ne dépend ni de Gemini ni de DuckDuckGo.
    client = TestClient(create_app(team=object()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_diffuse_la_trajectoire_puis_le_rapport():
    client = TestClient(create_app(FakeTeam(HAPPY)))

    response = client.post("/ask", json={"question": "question ?"})

    assert response.headers["content-type"].startswith("text/event-stream")
    events = events_of(response)
    assert [e["step"] for e in events] == ["research", "writer", "verifier", "done"]
    assert events[-1] == {
        "step": "done",
        "status": "approved",
        "report": "Rapport [https://a.com]",
        "revisions": 0,
    }


def test_ask_n_envoie_jamais_le_texte_brut_des_pages():
    client = TestClient(create_app(FakeTeam(HAPPY)))

    response = client.post("/ask", json={"question": "question ?"})

    assert "TEXTE BRUT" not in response.text
    assert events_of(response)[0]["sources"] == [{"url": "https://a.com", "title": "A"}]


def test_question_invalide_rejetee_sans_lancer_l_equipe():
    team = FakeTeam(HAPPY)
    client = TestClient(create_app(team))

    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert team.calls == 0  # aucun quota dépensé


def test_panne_en_cours_de_route_devient_un_evenement_error(caplog):
    secret = RuntimeError("quota du projet my-project-123, key=AIza-SECRET")
    client = TestClient(create_app(FakeTeam(HAPPY[:1], error=secret)))

    response = client.post("/ask", json={"question": "question ?"})

    events = events_of(response)
    assert [e["step"] for e in events] == ["research", "error"]
    assert events[-1]["message"] == "RuntimeError"
    # Le détail reste en cuisine : dans les journaux, jamais chez le client.
    assert "AIza-SECRET" not in response.text
    assert "AIza-SECRET" in caplog.text


def test_la_page_est_servie_et_n_interprete_jamais_le_html_du_llm():
    response = TestClient(create_app(team=object())).get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Garde-fou XSS : le rapport (texte d'un LLM nourri par le web) ne doit
    # jamais être interprété comme du HTML.
    assert "innerHTML" not in response.text
    assert 'fetch("/ask"' in response.text


def test_aucune_source_n_affiche_pas_de_coche_verte():
    # Bug trouvé en essai réel (DuckDuckGo en panne) : en interne approved=True,
    # mais l'utilisateur ne doit pas voir "toutes les affirmations sont sourcées".
    steps = [
        {"research": {"notes": "", "sources": [], "searches": 3}},
        {"no_sources": {"report": "Aucune source…", "verdict": Verdict(approved=True)}},
    ]
    client = TestClient(create_app(FakeTeam(steps)))

    events = events_of(client.post("/ask", json={"question": "question ?"}))

    assert [e["step"] for e in events] == ["research", "no_sources", "done"]
    assert events[-1]["status"] == "no_sources"


def test_les_titres_sont_nettoyes_des_sauts_de_ligne():
    # Titre réel renvoyé par Tavily pour aph.gov.au.
    src = {"url": "https://aph.gov.au", "title": "Creating\r\n        \r\n     – Parliament"}
    steps = [{"research": {"notes": "", "sources": [src], "searches": 1}}]
    client = TestClient(create_app(FakeTeam(steps)))

    events = events_of(client.post("/ask", json={"question": "question ?"}))

    assert events[0]["sources"][0]["title"] == "Creating – Parliament"
