# preprocessing.py - Split des données et générateurs d'images
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB

import os
import re
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


IMG_SIZE   = (64, 64)
BATCH_SIZE = 32

# Les noms de fichiers du jeu NIH commencent par un identifiant de cas,
# eventuellement suivi d'un identifiant de patient. Trois formes existent :
#   C100P61ThinF_IMG_...   C68P29N_ThinF_IMG_...   C1_thinF_IMG_...
# Le prefixe "C<n>" (avec "P<n>" s'il est present) identifie le frottis.
RE_PATIENT = re.compile(r"^(C\d+)(P\d+)?", re.IGNORECASE)


def extraire_patient(chemin: str) -> str:
    """Retourne l'identifiant de patient déduit du nom de fichier."""
    nom = os.path.basename(chemin)
    correspondance = RE_PATIENT.match(nom)
    return correspondance.group(0).upper() if correspondance else nom


def _make_df(paths, lbls) -> pd.DataFrame:
    return pd.DataFrame({
        "filename": paths,
        "class":    ["Parasitized" if l == 0 else "Uninfected" for l in lbls]
    })


def split_donnees(chemins: list, labels: list,
                  val_size: float = 0.15,
                  test_size: float = 0.15,
                  seed: int = 42,
                  par_patient: bool = False) -> tuple:
    """
    Divise le dataset en train / validation / test (70% / 15% / 15%).

    par_patient=False : découpage image par image, stratifié par classe.
        Attention, les cellules d'un même frottis se retrouvent alors des
        deux côtés du découpage. Sur ce jeu de données, 100 % des images
        de test proviennent de patients également vus à l'entraînement,
        ce qui rend l'estimation de performance optimiste.

    par_patient=True : découpage par patient. Aucun patient n'apparaît
        dans deux ensembles. C'est le protocole correct pour estimer la
        performance sur de nouveaux patients, et il donne des scores
        plus bas mais honnêtes.
    """
    if par_patient:
        return _split_par_patient(chemins, labels, val_size, test_size, seed)

    X_temp, X_test, y_temp, y_test = train_test_split(
        chemins, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_temp
    )

    print(f"  Split par image  ->  train : {len(X_train)}  |  "
          f"val : {len(X_val)}  |  test : {len(X_test)}")

    return _make_df(X_train, y_train), _make_df(X_val, y_val), _make_df(X_test, y_test)


def _split_par_patient(chemins: list, labels: list,
                       val_size: float, test_size: float, seed: int) -> tuple:
    """Découpe en gardant chaque patient entièrement dans un seul ensemble."""
    chemins = np.asarray(chemins)
    labels  = np.asarray(labels)
    groupes = np.asarray([extraire_patient(c) for c in chemins])

    separateur = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_temp, idx_test = next(separateur.split(chemins, labels, groups=groupes))

    val_ratio = val_size / (1 - test_size)
    separateur_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_tr, idx_val = next(separateur_val.split(
        chemins[idx_temp], labels[idx_temp], groups=groupes[idx_temp]))

    idx_train = idx_temp[idx_tr]
    idx_val   = idx_temp[idx_val]

    n_patients = len(set(groupes))
    print(f"  Split par patient  ->  train : {len(idx_train)}  |  "
          f"val : {len(idx_val)}  |  test : {len(idx_test)}")
    print(f"  Patients : {n_patients} au total, "
          f"{len(set(groupes[idx_train]))} / {len(set(groupes[idx_val]))} / "
          f"{len(set(groupes[idx_test]))} par ensemble")

    for nom, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        part = float((labels[idx] == 0).mean())
        print(f"    {nom:<6} {len(idx):>6} images, {part * 100:5.1f} % parasitees")

    return (_make_df(chemins[idx_train].tolist(), labels[idx_train].tolist()),
            _make_df(chemins[idx_val].tolist(),   labels[idx_val].tolist()),
            _make_df(chemins[idx_test].tolist(),  labels[idx_test].tolist()))


def creer_generateurs(df_train, df_val, df_test, seed: int = 42) -> tuple:
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
        seed=seed
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

    print(f"  Générateurs créés  ->  "
          f"train: {gen_train.n}  |  val: {gen_val.n}  |  test: {gen_test.n}")
    print(f"  Classes : {gen_train.class_indices}")

    return gen_train, gen_val, gen_test
