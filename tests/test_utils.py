# test_utils.py - Tests unitaires : préparation d'image et prédiction
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB
#
# Ces tests couvrent le chemin réellement utilisé par l'API en production,
# de l'image brute jusqu'au dictionnaire de prédiction.

import numpy as np
import pytest

from utils import preparer_image, predire_image
from models import build_cnn_simple


@pytest.fixture(scope="module")
def modele():
    """Un CNN non entraîné suffit : on teste le format de sortie, pas la justesse."""
    m = build_cnn_simple()
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


class TestPreparerImage:

    def test_forme_attendue(self, image_png):
        """L'image doit être redimensionnée au format d'entrée du réseau."""
        arr = preparer_image(image_png)
        assert arr.shape == (1, 64, 64, 3)

    def test_valeurs_normalisees(self, image_png):
        """Les pixels doivent être ramenés dans l'intervalle [0, 1]."""
        arr = preparer_image(image_png)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_type_float32(self, image_png):
        """Keras attend du float32, pas du float64."""
        assert preparer_image(image_png).dtype == np.float32

    def test_fichier_invalide(self, tmp_path):
        """Un fichier qui n'est pas une image doit lever une erreur, pas passer."""
        faux = tmp_path / "faux.png"
        faux.write_bytes(b"ceci n'est pas une image")
        with pytest.raises(Exception):
            preparer_image(str(faux))


class TestPredireImage:

    def test_cles_presentes(self, modele, image_png):
        """Le contrat de sortie consommé par l'API doit être respecté."""
        res = predire_image(modele, image_png)
        assert set(res) == {"label", "probabilite", "proba_parasite", "classe_id"}

    def test_label_coherent_avec_classe(self, modele, image_png):
        """Le label doit correspondre à l'identifiant de classe."""
        res = predire_image(modele, image_png)
        attendu = "Uninfected" if res["classe_id"] == 1 else "Parasitized"
        assert res["label"] == attendu

    def test_probabilites_dans_les_bornes(self, modele, image_png):
        """Les deux probabilités sont des pourcentages valides."""
        res = predire_image(modele, image_png)
        assert 0.0 <= res["probabilite"] <= 100.0
        assert 0.0 <= res["proba_parasite"] <= 100.0

    def test_confiance_toujours_majoritaire(self, modele, image_png):
        """La confiance porte sur la classe prédite, elle est donc >= 50 %."""
        res = predire_image(modele, image_png)
        assert res["probabilite"] >= 50.0

    def test_proba_parasite_coherente(self, modele, image_png):
        """Si le modèle prédit Parasitized, la probabilité de parasitisme domine."""
        res = predire_image(modele, image_png)
        if res["classe_id"] == 0:
            assert res["proba_parasite"] >= 50.0
        else:
            assert res["proba_parasite"] < 50.0
