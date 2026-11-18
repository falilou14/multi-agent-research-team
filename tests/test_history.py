import pytest

from multi_agent_research_team.history import History


@pytest.fixture
def history(tmp_path):
    return History(tmp_path / "history.db")


def save(history, visitor, question="q ?", status="approved"):
    return history.add(
        visitor,
        question=question,
        status=status,
        report="Rapport [https://a.com]",
        sources=[{"url": "https://a.com", "title": "A"}],
        revisions=1,
        duration_s=12.34,
    )


def test_une_recherche_enregistree_se_relit_entiere(history):
    search_id = save(history, "alice", question="Capitale de l'Australie ?")

    item = history.get("alice", search_id)

    assert item["question"] == "Capitale de l'Australie ?"
    assert item["report"] == "Rapport [https://a.com]"
    assert item["sources"] == [{"url": "https://a.com", "title": "A"}]
    assert item["revisions"] == 1 and item["duration_s"] == 12.3
    assert "visitor" not in item


def test_la_liste_est_la_plus_recente_d_abord_et_limitee(history):
    for i in range(5):
        save(history, "alice", question=f"q{i}")

    items = history.list("alice", limit=3)

    assert [i["question"] for i in items] == ["q4", "q3", "q2"]


def test_un_visiteur_ne_voit_jamais_l_historique_d_un_autre(history):
    # Faille IDOR : connaître (ou deviner) le numéro ne doit pas suffire.
    alice_id = save(history, "alice")

    assert history.get("bob", alice_id) is None
    assert history.list("bob") == []


def test_effacer_ne_supprime_que_son_propre_historique(history):
    save(history, "alice")
    save(history, "alice")
    bob_id = save(history, "bob")

    assert history.clear("alice") == 2
    assert history.list("alice") == []
    assert history.get("bob", bob_id) is not None


def test_l_historique_survit_a_un_redemarrage(tmp_path):
    search_id = save(History(tmp_path / "h.db"), "alice")

    # Nouvel objet = serveur redémarré : les données sont dans le fichier.
    assert History(tmp_path / "h.db").get("alice", search_id) is not None
