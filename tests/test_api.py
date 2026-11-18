import json

from fastapi.testclient import TestClient

from multi_agent_research_team.api import RateLimiter, create_app
from multi_agent_research_team.history import History
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
        "history_id": None,  # pas de base dans ce test
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


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_fenetre_glissante_refuse_puis_oublie_les_vieilles_questions():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=2, window_s=3600, clock=clock)

    assert limiter.check("ip") is None
    clock.now += 600
    assert limiter.check("ip") is None
    assert limiter.check("ip") == 3000  # libre quand la 1re (t=1000) expire
    assert limiter.check("autre-ip") is None  # chaque visiteur a son budget

    clock.now += 3000  # 1re question vieille d'une heure : oubliée
    assert limiter.check("ip") is None


def test_au_dela_du_plafond_429_sans_lancer_l_equipe():
    team = FakeTeam(HAPPY)
    client = TestClient(create_app(team, RateLimiter(max_requests=1, clock=FakeClock())))

    assert client.post("/ask", json={"question": "question ?"}).status_code == 200
    refused = client.post("/ask", json={"question": "question ?"})

    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "3600"
    assert team.calls == 1  # le refus n'a rien coûté


def test_la_recherche_est_enregistree_et_relisible(tmp_path):
    client = TestClient(create_app(FakeTeam(HAPPY), history=History(tmp_path / "h.db")))

    done = events_of(client.post("/ask", json={"question": "Capitale ?"}))[-1]
    items = client.get("/history").json()["items"]
    item = client.get(f"/history/{done['history_id']}").json()

    assert [i["question"] for i in items] == ["Capitale ?"]
    assert item["report"] == "Rapport [https://a.com]"
    assert item["sources"] == [{"url": "https://a.com", "title": "A"}]  # jamais le texte brut


def test_un_autre_navigateur_ne_voit_pas_mon_historique(tmp_path):
    history = History(tmp_path / "h.db")
    alice = TestClient(create_app(FakeTeam(HAPPY), history=history))
    bob = TestClient(create_app(FakeTeam(HAPPY), history=history))  # autres cookies

    search_id = events_of(alice.post("/ask", json={"question": "secret ?"}))[-1]["history_id"]

    assert bob.get("/history").json() == {"items": []}
    assert bob.get(f"/history/{search_id}").status_code == 404  # IDOR bloquée


def test_le_cookie_visiteur_est_httponly(tmp_path):
    client = TestClient(create_app(FakeTeam(HAPPY), history=History(tmp_path / "h.db")))

    response = client.post("/ask", json={"question": "q ?"})

    assert "httponly" in response.headers["set-cookie"].lower()


def test_un_cookie_trafique_est_ignore(tmp_path):
    history = History(tmp_path / "h.db")
    history.add("' OR 1=1 --", question="q", status="approved", report="r",
                sources=[], revisions=0, duration_s=1)  # fmt: skip
    client = TestClient(create_app(FakeTeam(HAPPY), history=history))
    client.cookies.set("visitor", "' OR 1=1 --")

    assert client.get("/history").json() == {"items": []}


def test_effacer_mon_historique(tmp_path):
    client = TestClient(create_app(FakeTeam(HAPPY), history=History(tmp_path / "h.db")))
    client.post("/ask", json={"question": "q ?"})

    assert client.delete("/history").status_code == 204
    assert client.get("/history").json() == {"items": []}


def test_une_base_en_panne_ne_fait_pas_perdre_la_reponse():
    class BrokenHistory:
        def add(self, *args, **kwargs):
            raise OSError("disque plein")

    client = TestClient(create_app(FakeTeam(HAPPY), history=BrokenHistory()))

    done = events_of(client.post("/ask", json={"question": "q ?"}))[-1]

    assert done["status"] == "approved" and done["history_id"] is None
