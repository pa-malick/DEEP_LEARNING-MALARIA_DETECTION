# ================================================================
# train.py  –  Entraînement des modèles CNN
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

import os
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


def entrainer_modele(nom: str, modele, gen_train, gen_val,
                     epochs: int = 20, models_dir: str = "models") -> dict:
    """
    Entraîne un modèle CNN avec des callbacks intelligents.

    Callbacks utilisés :
      - EarlyStopping       : arrêt si la val_loss ne s'améliore plus
      - ModelCheckpoint     : sauvegarde automatique du meilleur état
      - ReduceLROnPlateau   : réduit le learning rate si stagnation

    Paramètres
    ----------
    nom       : str  – nom du modèle (pour la sauvegarde)
    modele    : keras.Model
    gen_train : générateur d'entraînement
    gen_val   : générateur de validation
    epochs    : int  – nombre maximal d'époques
    models_dir: str  – dossier de sauvegarde des checkpoints

    Retourne
    --------
    history : dict  –  historique d'entraînement (loss, accuracy par époque)
    """
    os.makedirs(models_dir, exist_ok=True)
    chemin_checkpoint = os.path.join(models_dir, f"{nom}.keras")

    callbacks = [
        # Arrête l'entraînement si val_loss ne progresse plus pendant 5 époques
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Sauvegarde uniquement le meilleur état du modèle
        ModelCheckpoint(
            filepath=chemin_checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        ),
        # Divise le learning rate par 2 si val_loss stagne 3 époques
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n{'─' * 50}")
    print(f"  Entraînement : {nom}")
    print(f"  Époques max  : {epochs}  |  EarlyStopping patience=5")
    print(f"{'─' * 50}")

    history = modele.fit(
        gen_train,
        epochs=epochs,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n[✔] {nom} entraîné  →  sauvegardé dans {chemin_checkpoint}")
    return history.history


def entrainer_tous(modeles: dict, gen_train, gen_val,
                   epochs: int = 20) -> dict:
    """
    Lance l'entraînement séquentiel de tous les modèles.

    Paramètres
    ----------
    modeles   : dict  –  { nom: modele_compilé }
    gen_train : générateur d'entraînement
    gen_val   : générateur de validation
    epochs    : int   –  nombre max d'époques

    Retourne
    --------
    histories : dict  –  { nom: history }
    """
    print("\n┌─ ENTRAÎNEMENT DES 3 CNN ──────────────────────────────┐")
    histories = {}

    for nom, modele in modeles.items():
        histories[nom] = entrainer_modele(nom, modele, gen_train, gen_val, epochs)

    print("\n└─ Tous les modèles ont été entraînés ──────────────────┘\n")
    return histories
