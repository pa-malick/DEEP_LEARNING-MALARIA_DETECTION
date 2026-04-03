# ================================================================
# utils.py  –  Sauvegarde, chargement et prédiction unitaire
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


IMG_SIZE = (64, 64)


def sauvegarder_meilleur(modele, nom: str,
                          models_dir: str = "models") -> None:
    """
    Sauvegarde le meilleur modèle dans models/best_model.keras
    """
    os.makedirs(models_dir, exist_ok=True)
    chemin = os.path.join(models_dir, "best_model.keras")
    modele.save(chemin)
    print(f"[✔] Meilleur modèle sauvegardé : {chemin}  ({nom})")


def charger_meilleur(models_dir: str = "models"):
    """
    Charge le meilleur modèle depuis models/best_model.keras

    Retourne
    --------
    tf.keras.Model
    """
    chemin = os.path.join(models_dir, "best_model.keras")
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Modèle introuvable : '{chemin}'\n"
            "→ Lancez d'abord :  python main.py"
        )
    modele = tf.keras.models.load_model(chemin)
    print(f"[✔] Modèle chargé : {chemin}")
    return modele


def charger_metriques(chemin: str = "metrics/results.json") -> dict:
    if not os.path.exists(chemin):
        return {}
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)


def preparer_image(image_path: str) -> np.ndarray:
    """
    Prépare une image pour la prédiction :
      - Redimensionne à 64×64
      - Convertit en RGB
      - Normalise les pixels dans [0, 1]
      - Ajoute la dimension batch

    Retourne
    --------
    np.ndarray de forme (1, 64, 64, 3)
    """
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 64, 64, 3)


def predire_image(modele, image_path: str) -> dict:
    """
    Effectue une prédiction sur une seule image.

    Retourne
    --------
    dict  –  { label, probabilite, classe_id }
    """
    arr    = preparer_image(image_path)
    proba  = float(modele.predict(arr, verbose=0)[0][0])

    # Correspondance : 0=Parasitized, 1=Uninfected
    # (selon class_indices de ImageDataGenerator)
    classe_id = 1 if proba > 0.5 else 0
    label     = "Uninfected" if classe_id == 1 else "Parasitized"
    proba_aff = proba if classe_id == 1 else 1 - proba

    return {
        "label"      : label,
        "probabilite": round(proba_aff * 100, 2),
        "classe_id"  : classe_id
    }
