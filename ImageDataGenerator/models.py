# ================================================================
# models.py  –  Les 3 architectures CNN
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# On implémente 3 réseaux convolutifs de complexité croissante :
#
#   CNN_Simple  – architecture basique, bonne baseline
#   CNN_Deep    – plus de couches, capture des features complexes
#   CNN_BN      – avec Batch Normalization + Dropout, plus régularisé
#
# Les 3 partagent la même taille d'entrée (64×64×3) et la même
# sortie (sigmoïde pour classification binaire).
# ================================================================

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)


IMG_SIZE = (64, 64, 3)   # hauteur × largeur × canaux RGB


# ── Modèle 1 : CNN Simple ────────────────────────────────────────

def build_cnn_simple() -> Sequential:
    """
    Architecture CNN basique :
      Conv → Pool → Conv → Pool → Flatten → Dense → Sortie

    C'est le point de départ classique pour les tâches de vision.
    Peu de paramètres, entraînement rapide, utile comme baseline.
    """
    model = Sequential(name="CNN_Simple", layers=[
        Input(shape=IMG_SIZE),

        # Bloc 1 : détection de features bas niveau (bords, textures)
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Bloc 2 : features de niveau intermédiaire
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Classifieur
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")   # 1 neurone → proba [0,1]
    ])
    return model


# ── Modèle 2 : CNN Deep ──────────────────────────────────────────

def build_cnn_deep() -> Sequential:
    """
    Architecture CNN plus profonde :
      3 blocs Conv → Pool + tête Dense plus large

    Plus de couches = capacité à apprendre des représentations
    plus abstraites et discriminantes des images de cellules.
    Le risque est un léger surapprentissage sur de petits datasets.
    """
    model = Sequential(name="CNN_Deep", layers=[
        Input(shape=IMG_SIZE),

        # Bloc 1
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Bloc 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Bloc 3 : features abstraites (formes, patterns complexes)
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Classifieur élargi
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model


# ── Modèle 3 : CNN + Batch Normalization ────────────────────────

def build_cnn_bn() -> Sequential:
    """
    Architecture CNN avec Batch Normalization après chaque bloc conv.

    La BatchNormalization normalise les activations intermédiaires,
    ce qui accélère la convergence, stabilise l'entraînement et
    agit comme un léger régularisateur.

    C'est généralement l'architecture la plus robuste des trois.
    """
    model = Sequential(name="CNN_BN", layers=[
        Input(shape=IMG_SIZE),

        # Bloc 1 : Conv + BN + Pool
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Bloc 2 : Conv + BN + Pool
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Bloc 3 : Conv + BN + Pool
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classifieur
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model


# ── Factory : récupérer tous les modèles ────────────────────────

def get_modeles() -> dict:
    """
    Instancie et compile les 3 modèles CNN.

    On utilise :
      - Adam comme optimiseur (adaptatif, généralement le meilleur choix)
      - binary_crossentropy comme loss (classification binaire)
      - accuracy comme métrique principale

    Retourne
    --------
    dict  –  { nom: modele_compilé }
    """
    modeles = {
        "CNN_Simple" : build_cnn_simple(),
        "CNN_Deep"   : build_cnn_deep(),
        "CNN_BN"     : build_cnn_bn(),
    }

    for nom, modele in modeles.items():
        modele.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print(f"[✔] {nom} compilé  –  {modele.count_params():,} paramètres")

    return modeles
