# ================================================================
# test_models.py  –  Tests unitaires : architectures CNN
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
# ================================================================

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ImageDataGenerator.models import get_modeles, build_cnn_simple, build_cnn_deep, build_cnn_bn

IMG_SHAPE = (1, 64, 64, 3)   # batch de 1 image 64×64 RGB


class TestArchitectures:

    def test_trois_modeles(self):
        modeles = get_modeles()
        assert len(modeles) == 3

    def test_noms_modeles(self):
        noms = set(get_modeles().keys())
        assert noms == {"CNN_Simple", "CNN_Deep", "CNN_BN"}

    def test_forme_sortie_cnn_simple(self):
        m   = build_cnn_simple()
        m.compile(optimizer="adam", loss="binary_crossentropy")
        x   = np.random.rand(*IMG_SHAPE).astype(np.float32)
        out = m.predict(x, verbose=0)
        assert out.shape == (1, 1), "La sortie doit être (batch, 1)"
        assert 0 <= float(out[0][0]) <= 1, "La sortie doit être dans [0,1]"

    def test_forme_sortie_cnn_deep(self):
        m   = build_cnn_deep()
        m.compile(optimizer="adam", loss="binary_crossentropy")
        x   = np.random.rand(*IMG_SHAPE).astype(np.float32)
        out = m.predict(x, verbose=0)
        assert out.shape == (1, 1)
        assert 0 <= float(out[0][0]) <= 1

    def test_forme_sortie_cnn_bn(self):
        m   = build_cnn_bn()
        m.compile(optimizer="adam", loss="binary_crossentropy")
        x   = np.random.rand(*IMG_SHAPE).astype(np.float32)
        out = m.predict(x, verbose=0)
        assert out.shape == (1, 1)
        assert 0 <= float(out[0][0]) <= 1

    def test_modeles_compiles(self):
        """Vérifie que tous les modèles sont bien compilés (ont un optimizer)."""
        for nom, m in get_modeles().items():
            assert m.optimizer is not None, f"{nom} : pas d'optimizer"

    def test_nombre_parametres_croissant(self):
        """CNN_Deep doit avoir plus de paramètres que CNN_Simple."""
        m_simple = build_cnn_simple()
        m_simple.compile(optimizer="adam", loss="binary_crossentropy")
        m_deep   = build_cnn_deep()
        m_deep.compile(optimizer="adam", loss="binary_crossentropy")
        assert m_deep.count_params() > m_simple.count_params()
