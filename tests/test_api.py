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


def test_panne_en_cours_de_route_devient_un_evenement_error():
    client = TestClient(create_app(FakeTeam(HAPPY[:1], error=RuntimeError("boom"))))

    events = events_of(client.post("/ask", json={"question": "question ?"}))

    assert [e["step"] for e in events] == ["research", "error"]
    assert events[-1]["message"] == "RuntimeError"
