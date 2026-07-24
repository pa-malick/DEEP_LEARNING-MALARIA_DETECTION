# conftest.py - Fixtures et chemins d'import partagés
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RACINE, "src"))
sys.path.insert(0, os.path.join(RACINE, "app"))


@pytest.fixture
def image_png(tmp_path):
    """Crée une image PNG valide sur disque et retourne son chemin."""
    chemin = tmp_path / "cellule.png"
    tableau = np.random.randint(0, 256, (120, 130, 3), dtype=np.uint8)
    Image.fromarray(tableau).save(chemin)
    return str(chemin)


@pytest.fixture
def octets_png():
    """Retourne les octets d'une image PNG valide, pour un upload."""
    tampon = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    flux = io.BytesIO()
    Image.fromarray(tampon).save(flux, format="PNG")
    return flux.getvalue()


@pytest.fixture(scope="module")
def client():
    """Client de test Flask. Fonctionne même si aucun modèle n'est entraîné."""
    import app as application
    application.app.config["TESTING"] = True
    with application.app.test_client() as c:
        yield c
