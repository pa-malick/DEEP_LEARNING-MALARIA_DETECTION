# models.py – Les 3 architectures CNN comparées
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)


IMG_SIZE = (64, 64, 3)


def build_cnn_simple() -> Sequential:
    """CNN de base avec 2 blocs convolutifs. Sert de point de comparaison."""
    model = Sequential(name="CNN_Simple", layers=[
        Input(shape=IMG_SIZE),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model


def build_cnn_deep() -> Sequential:
    """CNN plus profond avec 3 blocs convolutifs pour capturer des features plus complexes."""
    model = Sequential(name="CNN_Deep", layers=[
        Input(shape=IMG_SIZE),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model


def build_cnn_bn() -> Sequential:
    """CNN avec BatchNormalization après chaque bloc pour un entraînement plus stable."""
    model = Sequential(name="CNN_BN", layers=[
        Input(shape=IMG_SIZE),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model


def get_modeles() -> dict:
    """Crée et compile les 3 modèles. Retourne un dict { nom: modele }."""
    modeles = {
        "CNN_Simple": build_cnn_simple(),
        "CNN_Deep":   build_cnn_deep(),
        "CNN_BN":     build_cnn_bn(),
    }

    for nom, modele in modeles.items():
        modele.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print(f"  {nom} compilé  –  {modele.count_params():,} paramètres")

    return modeles
