"""
Le jeu d'évaluation est FIGÉ (Ch.5) : on mesure chaque changement du système
sur le même "circuit". Modifier les questions après avoir vu des scores
reviendrait à ajuster le système au test (piège déjà vu avec recall@k dans
le projet assurance).

Le gel est imposé par une empreinte : toute modification fait échouer ce
test. Pour changer le jeu volontairement, il faut mettre à jour
FROZEN_FINGERPRINT dans le même commit -- un acte visible dans l'historique,
jamais un ajustement discret.
"""

import hashlib
import json
from pathlib import Path

DATASET = Path(__file__).parent.parent / "evals" / "questions.json"
FAMILIES = {
    "factuelle_simple",
    "plusieurs_faits",
    "chiffre_precis",
    "premisse_fausse",
    "introuvable",
    "recent",
}
FROZEN_FINGERPRINT = "137d8f81829ff7758edfe9ae96209ecbacc6796dd18e6079836dcb54f2e405bc"


def load():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_le_jeu_d_evaluation_est_fige():
    # Empreinte du CONTENU (JSON canonique), pas des octets du fichier : sinon
    # les fins de ligne Windows (CRLF) vs Linux (LF) en CI donneraient deux
    # empreintes différentes pour les mêmes questions.
    canonical = json.dumps(load(), sort_keys=True, ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert fingerprint == FROZEN_FINGERPRINT, (
        "evals/questions.json a changé. Le jeu est figé : si c'est voulu, mets à jour "
        "FROZEN_FINGERPRINT dans le même commit et explique pourquoi."
    )


def test_chaque_question_est_bien_formee():
    questions = load()
    ids = [q["id"] for q in questions]

    assert len(ids) == len(set(ids)), "identifiants en double"
    for q in questions:
        assert set(q) == {"id", "family", "question", "expected_behavior", "must_mention"}, q["id"]
        assert q["family"] in FAMILIES, q["id"]
        assert q["question"].strip() and q["expected_behavior"].strip(), q["id"]
    assert {q["family"] for q in questions} == FAMILIES  # chaque famille est couverte


# --- Jeu v2 (18 nov.) : NOUVELLE version, le v1 reste figé tel quel ci-dessus.
# Les chiffres mesurés sur v1 ne valent que pour v1 ; v2 a sa propre empreinte.
DATASET_V2_DEV = DATASET.parent / "questions_v2_dev.json"
FAMILIES_V2 = FAMILIES | {"opinion"}  # questions subjectives (vu en réel : "meilleur joueur")
FROZEN_FINGERPRINT_V2_DEV = "03d578ae20d9bcbe8cc69a2a5231c85fb5b12b423f38bf504b638a5802df357b"


def load_v2_dev():
    return json.loads(DATASET_V2_DEV.read_text(encoding="utf-8"))


def test_le_jeu_v2_dev_est_fige():
    canonical = json.dumps(load_v2_dev(), sort_keys=True, ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert fingerprint == FROZEN_FINGERPRINT_V2_DEV, (
        "evals/questions_v2_dev.json a changé : si c'est voulu, mets à jour "
        "FROZEN_FINGERPRINT_V2_DEV dans le même commit et explique pourquoi."
    )


def test_v2_dev_contient_les_16_questions_v1_inchangees():
    # Continuité : on peut comparer v1 et v2 sur les 16 questions communes.
    v2 = {q["id"]: q for q in load_v2_dev()}
    for q in load():
        assert v2[q["id"]] == q, q["id"]


def test_chaque_question_v2_est_bien_formee():
    questions = load_v2_dev()
    ids = [q["id"] for q in questions]

    assert len(questions) == 30
    assert len(ids) == len(set(ids)), "identifiants en double"
    for q in questions:
        assert set(q) == {"id", "family", "question", "expected_behavior", "must_mention"}, q["id"]
        assert q["family"] in FAMILIES_V2, q["id"]
    assert {q["family"] for q in questions} == FAMILIES_V2
