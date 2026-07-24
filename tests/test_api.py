# test_api.py - Tests d'intégration : endpoints Flask
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB
#
# Ces tests s'exécutent avec ou sans modèle entraîné : c'est ce qui permet
# de les lancer en intégration continue, où models/ est vide.

import io
import os


def _upload(octets, nom="cellule.png"):
    return {"image": (io.BytesIO(octets), nom)}


class TestEndpointsSimples:

    def test_health_repond(self, client):
        """/health doit toujours répondre, même sans modèle chargé."""
        r = client.get("/health")
        assert r.status_code == 200
        assert "modele_charge" in r.get_json()

    def test_metrics_est_du_json(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert isinstance(r.get_json(), dict)

    def test_page_accueil(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_avertissement_medical_present(self, client):
        """La page doit porter la mention indiquant que ce n'est pas un diagnostic."""
        contenu = client.get("/").get_data(as_text=True)
        assert "pas un dispositif médical" in contenu


class TestValidationUpload:
    """La validation doit répondre 400 indépendamment de l'état du modèle."""

    def test_sans_fichier(self, client):
        r = client.post("/predict", data={}, content_type="multipart/form-data")
        assert r.status_code == 400
        assert "erreur" in r.get_json()

    def test_nom_de_fichier_vide(self, client, octets_png):
        r = client.post("/predict", data=_upload(octets_png, ""),
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_extension_refusee(self, client, octets_png):
        r = client.post("/predict", data=_upload(octets_png, "script.exe"),
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_fichier_trop_gros(self, client):
        """Au-dela de 8 Mo, l'API doit refuser sans tenter de lire l'image."""
        r = client.post("/predict", data=_upload(b"\x00" * (9 * 1024 * 1024)),
                        content_type="multipart/form-data")
        assert r.status_code == 413

    def test_contenu_non_image(self, client):
        """Un fichier texte renomme .png ne doit pas provoquer d'erreur 500."""
        r = client.post("/predict", data=_upload(b"pas une image"),
                        content_type="multipart/form-data")
        assert r.status_code in (400, 503)
        assert r.status_code != 500

    def test_message_erreur_sans_detail_interne(self, client):
        """Le message renvoye ne doit pas exposer de chemin serveur."""
        r = client.post("/predict", data=_upload(b"pas une image"),
                        content_type="multipart/form-data")
        message = r.get_json().get("erreur", "")
        assert "Traceback" not in message
        assert os.sep not in message


class TestPrediction:

    def test_prediction_ou_service_indisponible(self, client, octets_png):
        """
        Avec un modele : reponse 200 au format attendu.
        Sans modele : 503 explicite. Jamais de 500.
        """
        r = client.post("/predict", data=_upload(octets_png),
                        content_type="multipart/form-data")
        assert r.status_code in (200, 503)

        if r.status_code == 200:
            data = r.get_json()
            assert data["label"] in ("Parasitized", "Uninfected")
            assert data["classe_id"] in (0, 1)
            assert 0 <= data["probabilite"] <= 100

    def test_image_non_conservee_sur_le_serveur(self, client, octets_png):
        """L'image uploadee doit etre supprimee apres traitement."""
        import app as application
        client.post("/predict", data=_upload(octets_png),
                    content_type="multipart/form-data")
        restants = [f for f in os.listdir(application.UPLOAD_DIR)
                    if not f.startswith(".")]
        assert restants == []
