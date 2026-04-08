# train.py – Entraînement des modèles CNN
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

import os
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


def entrainer_modele(nom: str, modele, gen_train, gen_val,
                     epochs: int = 20,
                     models_dir: str = "models") -> dict:
    """Entraîne un modèle et retourne l'historique loss/accuracy."""
    os.makedirs(models_dir, exist_ok=True)
    chemin_checkpoint = os.path.join(models_dir, f"{nom}.keras")

    callbacks = [
        # Arrêt si val_loss ne s'améliore plus pendant 5 époques
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Sauvegarde du meilleur état selon val_accuracy
        ModelCheckpoint(
            filepath=chemin_checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        ),
        # Réduction du learning rate (factor=0.5) si stagnation pendant 3 époques
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

    print(f"\n  {nom} entraîné  →  sauvegardé dans {chemin_checkpoint}")
    return history.history


def entrainer_tous(modeles: dict, gen_train, gen_val,
                   epochs: int = 20) -> dict:
    """Lance l'entraînement séquentiel de tous les modèles."""
    print("\n┌─ ENTRAÎNEMENT DES 3 CNN ──────────────────────────────┐")
    histories = {}

    for nom, modele in modeles.items():
        histories[nom] = entrainer_modele(nom, modele, gen_train, gen_val, epochs)

    print("\n└─ Tous les modèles ont été entraînés ──────────────────┘\n")
    return histories
