# test_preprocessing.py – Tests unitaires : preprocessing
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import split_donnees


@pytest.fixture
def donnees_fictives():
    """200 chemins et labels fictifs (100 par classe) pour les tests."""
    chemins = (
        [f"/fake/Parasitized/img_{i}.png" for i in range(100)] +
        [f"/fake/Uninfected/img_{i}.png"  for i in range(100)]
    )
    labels = [0] * 100 + [1] * 100
    return chemins, labels


class TestSplitDonnees:

    def test_tailles_coherentes(self, donnees_fictives):
        """La somme des 3 ensembles doit égaler le total initial."""
        chemins, labels = donnees_fictives
        df_tr, df_val, df_te = split_donnees(chemins, labels)
        assert len(df_tr) + len(df_val) + len(df_te) == len(chemins)

    def test_proportion_test_approx(self, donnees_fictives):
        """Le set de test doit représenter environ 15% (±5% de tolérance)."""
        chemins, labels = donnees_fictives
        df_tr, df_val, df_te = split_donnees(chemins, labels, test_size=0.15)
        ratio = len(df_te) / len(chemins)
        assert abs(ratio - 0.15) < 0.05, f"Proportion test incorrecte : {ratio:.2f}"

    def test_colonnes_presentes(self, donnees_fictives):
        """Chaque DataFrame doit avoir les colonnes 'filename' et 'class'."""
        chemins, labels = donnees_fictives
        df_tr, df_val, df_te = split_donnees(chemins, labels)
        for df in [df_tr, df_val, df_te]:
            assert "filename" in df.columns
            assert "class"    in df.columns

    def test_labels_valides(self, donnees_fictives):
        """Les labels doivent être uniquement 'Parasitized' ou 'Uninfected'."""
        chemins, labels = donnees_fictives
        df_tr, df_val, df_te = split_donnees(chemins, labels)
        valeurs_valides = {"Parasitized", "Uninfected"}
        for df in [df_tr, df_val, df_te]:
            assert set(df["class"].unique()).issubset(valeurs_valides)

    def test_pas_de_chevauchement(self, donnees_fictives):
        """Train, val et test ne doivent pas partager d'images."""
        chemins, labels = donnees_fictives
        df_tr, df_val, df_te = split_donnees(chemins, labels)

        set_tr  = set(df_tr["filename"])
        set_val = set(df_val["filename"])
        set_te  = set(df_te["filename"])

        assert len(set_tr & set_val) == 0, "Chevauchement train/val détecté"
        assert len(set_tr & set_te)  == 0, "Chevauchement train/test détecté"
        assert len(set_val & set_te) == 0, "Chevauchement val/test détecté"
