# preprocessing.py – Split des données et générateurs d'images
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


IMG_SIZE   = (64, 64)
BATCH_SIZE = 32


def split_donnees(chemins: list, labels: list,
                  val_size: float = 0.15,
                  test_size: float = 0.15) -> tuple:
    """
    Divise le dataset en train / validation / test (70% / 15% / 15%).
    Le stratify garantit la même proportion de classes dans chaque partie.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        chemins, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=42,
        stratify=y_temp
    )

    print(f"  Split  →  train : {len(X_train)}  |  val : {len(X_val)}  |  test : {len(X_test)}")

    def make_df(paths, lbls):
        return pd.DataFrame({
            "filename": paths,
            "class":    ["Parasitized" if l == 0 else "Uninfected" for l in lbls]
        })

    return make_df(X_train, y_train), make_df(X_val, y_val), make_df(X_test, y_test)


def creer_generateurs(df_train, df_val, df_test) -> tuple:
    """
    Crée les générateurs Keras pour les trois splits.
    L'augmentation (rotation, flip, zoom) est appliquée uniquement sur le train
    pour enrichir les données sans modifier l'évaluation.
    """
    gen_train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest"
    )

    gen_eval = ImageDataGenerator(rescale=1.0 / 255)

    gen_train = gen_train_aug.flow_from_dataframe(
        df_train,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
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

    print(f"  Générateurs créés  →  "
          f"train: {gen_train.n}  |  val: {gen_val.n}  |  test: {gen_test.n}")
    print(f"  Classes : {gen_train.class_indices}")

    return gen_train, gen_val, gen_test
