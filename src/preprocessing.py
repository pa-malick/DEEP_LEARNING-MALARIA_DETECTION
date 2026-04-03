# ================================================================
# preprocessing.py  –  Prétraitement et générateurs d'images
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# On utilise ImageDataGenerator de Keras pour :
#   1. Normaliser les pixels (÷ 255 pour ramener dans [0,1])
#   2. Augmenter les données d'entraînement (rotation, flip…)
#   3. Générer les batchs à la volée (économise la RAM)
# ================================================================

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


# Taille cible de toutes les images (CNN attend une taille fixe)
IMG_SIZE   = (64, 64)
BATCH_SIZE = 32


def split_donnees(chemins: list, labels: list,
                  val_size: float = 0.15,
                  test_size: float = 0.15) -> tuple:
    """
    Divise le dataset en 3 parties : train / validation / test.

    Stratégie :
      1. Séparer 15% pour le test
      2. Du reste, séparer ~17.6% pour la validation (≈15% du total)
      3. Le reste (≈70%) va en entraînement

    Paramètres
    ----------
    chemins  : list  – chemins des images
    labels   : list  – étiquettes (0 ou 1)
    val_size : float – proportion de validation (du total)
    test_size: float – proportion de test (du total)

    Retourne
    --------
    df_train, df_val, df_test : pd.DataFrame avec colonnes [filename, class]
    """
    # Étape 1 : séparation test
    X_temp, X_test, y_temp, y_test = train_test_split(
        chemins, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    # Étape 2 : séparation val depuis le reste
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=42,
        stratify=y_temp
    )

    print(f"[✔] Split  →  train : {len(X_train)}  |  val : {len(X_val)}  |  test : {len(X_test)}")

    # On retourne des DataFrames avec le chemin et le label en string
    def make_df(paths, lbls):
        return pd.DataFrame({
            "filename": paths,
            "class":    ["Parasitized" if l == 0 else "Uninfected" for l in lbls]
        })

    return make_df(X_train, y_train), make_df(X_val, y_val), make_df(X_test, y_test)


def creer_generateurs(df_train, df_val, df_test) -> tuple:
    """
    Crée les trois générateurs Keras à partir des DataFrames.

    - Le générateur d'entraînement applique de l'augmentation de données
      pour améliorer la généralisation du modèle.
    - Les générateurs val/test ne font que normaliser (pas d'augmentation).

    Retourne
    --------
    gen_train, gen_val, gen_test : générateurs Keras prêts à l'emploi
    """

    # ── Générateur train : normalisation + augmentation ───────────
    # L'augmentation artificielle diversifie le dataset pour éviter
    # le surapprentissage (overfitting)
    gen_train_aug = ImageDataGenerator(
        rescale=1.0 / 255,          # normalisation : pixels dans [0, 1]
        rotation_range=20,           # rotation aléatoire ±20°
        width_shift_range=0.1,       # décalage horizontal ±10%
        height_shift_range=0.1,      # décalage vertical ±10%
        horizontal_flip=True,        # miroir horizontal aléatoire
        zoom_range=0.1,              # zoom aléatoire ±10%
        fill_mode="nearest"          # remplit les pixels manquants
    )

    # ── Générateurs val/test : normalisation uniquement ───────────
    gen_eval = ImageDataGenerator(rescale=1.0 / 255)

    # Création des flux depuis les DataFrames
    gen_train = gen_train_aug.flow_from_dataframe(
        df_train,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",         # sortie 0 ou 1
        shuffle=True,
        seed=42
    )

    gen_val = gen_eval.flow_from_dataframe(
        df_val,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    gen_test = gen_eval.flow_from_dataframe(
        df_test,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    print(f"[✔] Générateurs créés  →  "
          f"train: {gen_train.n}  |  val: {gen_val.n}  |  test: {gen_test.n}")
    print(f"     Correspondance classes : {gen_train.class_indices}")

    return gen_train, gen_val, gen_test
