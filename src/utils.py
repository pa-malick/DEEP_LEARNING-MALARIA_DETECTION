# utils.py - Sauvegarde, chargement et prédiction sur une image
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


IMG_SIZE = (64, 64)


def sauvegarder_meilleur(modele, nom: str, models_dir: str = "models") -> None:
    os.makedirs(models_dir, exist_ok=True)
    chemin = os.path.join(models_dir, "best_model.keras")
    modele.save(chemin)
    print(f"  Modele sauvegarde : {chemin}")


def charger_meilleur(models_dir: str = "models"):
    """Charge le meilleur modèle depuis le dossier models/."""
    chemin = os.path.join(models_dir, "best_model.keras")

    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Modèle introuvable dans '{models_dir}'\n"
            "Lancez d'abord :  python main.py"
        )

    modele = tf.keras.models.load_model(chemin)
    print("[OK] Modele charge")
    return modele


def charger_metriques(chemin: str = "metrics/results.json") -> dict:
    if not os.path.exists(chemin):
        return {}
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)


def preparer_image(image_path: str) -> np.ndarray:
    """Ouvre, redimensionne et normalise une image pour la prédiction."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predire_image(modele, image_path: str) -> dict:
    """
    Retourne la prédiction pour une image de cellule.

    La sortie du modèle est P(Uninfected), la classe 1 étant Uninfected.
    On expose donc deux valeurs distinctes, souvent confondues :
      probabilite     : confiance du modèle dans la classe qu'il a prédite
      proba_parasite  : probabilité que la cellule soit parasitée
    """
    arr            = preparer_image(image_path)
    proba_sain     = float(modele.predict(arr, verbose=0)[0][0])
    proba_parasite = 1.0 - proba_sain

    classe_id = 1 if proba_sain > 0.5 else 0
    label     = "Uninfected" if classe_id == 1 else "Parasitized"
    confiance = proba_sain if classe_id == 1 else proba_parasite

    return {
        "label"         : label,
        "probabilite"   : round(confiance * 100, 2),
        "proba_parasite": round(proba_parasite * 100, 2),
        "classe_id"     : classe_id
    }
