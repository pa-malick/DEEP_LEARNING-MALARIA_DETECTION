# test_preprocessing.py - Tests unitaires : preprocessing
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB

import pytest

from preprocessing import split_donnees, extraire_patient


@pytest.fixture
def donnees_fictives():
    """200 chemins et labels fictifs (100 par classe) pour les tests."""
    chemins = (
        [f"/fake/Parasitized/img_{i}.png" for i in range(100)] +
        [f"/fake/Uninfected/img_{i}.png"  for i in range(100)]
    )
    labels = [0] * 100 + [1] * 100
    return chemins, labels


@pytest.fixture
def donnees_patients():
    """
    40 patients, 20 cellules chacun, aux trois formats de nommage du jeu NIH.
    Chaque patient porte les deux classes.
    """
    chemins, labels = [], []
    for p in range(40):
        if p % 3 == 0:
            prefixe = f"C{p}P{p + 60}ThinF"
        elif p % 3 == 1:
            prefixe = f"C{p}_thinF"
        else:
            prefixe = f"C{p}ThinF"
        for c in range(20):
            classe = "Parasitized" if c < 10 else "Uninfected"
            chemins.append(f"/fake/{classe}/{prefixe}_IMG_20150604_1047_cell_{c}.png")
            labels.append(0 if c < 10 else 1)
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


class TestExtrairePatient:

    @pytest.mark.parametrize("nom, attendu", [
        ("C100P61ThinF_IMG_20150918_144104_cell_162.png", "C100P61"),
        ("C68P29N_ThinF_IMG_20150819_134712_cell_67.png",  "C68P29"),
        ("C1_thinF_IMG_20150604_104722_cell_115.png",      "C1"),
        ("C210ThinF_IMG_20151029_162357_cell_195.png",     "C210"),
        ("C48P9thinF_IMG_20150721_162732_cell_28.png",     "C48P9"),
    ])
    def test_formats_de_nommage(self, nom, attendu):
        """Les trois formats de nommage du jeu NIH doivent être reconnus."""
        assert extraire_patient(f"/data/Parasitized/{nom}") == attendu

    def test_cellules_du_meme_frottis_regroupees(self):
        """Deux cellules du même frottis doivent donner le même identifiant."""
        a = "/d/C116P77ThinF_IMG_20150930_171219_cell_110.png"
        b = "/d/C116P77ThinF_IMG_20150930_171954_cell_87.png"
        assert extraire_patient(a) == extraire_patient(b)

    def test_patients_differents_non_confondus(self):
        a = "/d/C116P77ThinF_IMG_20150930_171219_cell_110.png"
        b = "/d/C117P78ThinF_IMG_20150930_214317_cell_102.png"
        assert extraire_patient(a) != extraire_patient(b)


class TestSplitParPatient:

    def test_aucun_patient_partage(self, donnees_patients):
        """
        C'est la garantie centrale du split par patient : aucun patient ne
        doit apparaître dans deux ensembles, sinon l'évaluation est faussée.
        """
        chemins, labels = donnees_patients
        tr, va, te = split_donnees(chemins, labels, par_patient=True)

        p_tr = {extraire_patient(f) for f in tr["filename"]}
        p_va = {extraire_patient(f) for f in va["filename"]}
        p_te = {extraire_patient(f) for f in te["filename"]}

        assert p_tr & p_te == set(), "Patient partagé entre train et test"
        assert p_tr & p_va == set(), "Patient partagé entre train et val"
        assert p_va & p_te == set(), "Patient partagé entre val et test"

    def test_aucune_image_perdue(self, donnees_patients):
        chemins, labels = donnees_patients
        tr, va, te = split_donnees(chemins, labels, par_patient=True)
        assert len(tr) + len(va) + len(te) == len(chemins)

    def test_les_trois_ensembles_sont_non_vides(self, donnees_patients):
        chemins, labels = donnees_patients
        tr, va, te = split_donnees(chemins, labels, par_patient=True)
        assert len(tr) > 0 and len(va) > 0 and len(te) > 0

    def test_reproductible(self, donnees_patients):
        """Une même graine doit produire exactement le même découpage."""
        chemins, labels = donnees_patients
        a = split_donnees(chemins, labels, par_patient=True, seed=7)
        b = split_donnees(chemins, labels, par_patient=True, seed=7)
        for df_a, df_b in zip(a, b):
            assert list(df_a["filename"]) == list(df_b["filename"])

    def test_split_par_image_fuit_bien(self, donnees_patients):
        """
        Test de contraste : le split par image, lui, partage des patients.
        Il documente précisément le problème que le split par patient corrige.
        """
        chemins, labels = donnees_patients
        tr, _, te = split_donnees(chemins, labels, par_patient=False)
        p_tr = {extraire_patient(f) for f in tr["filename"]}
        p_te = {extraire_patient(f) for f in te["filename"]}
        assert p_tr & p_te, "Le split par image devrait partager des patients"
