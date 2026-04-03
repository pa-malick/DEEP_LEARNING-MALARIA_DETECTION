# ================================================================
# data_loader.py  –  Chargement et visualisation des images
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# Le dataset cell_images contient deux dossiers :
#   - Parasitized/   → cellules infectées par le paludisme
#   - Uninfected/    → cellules saines
# ================================================================

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# Noms des deux classes (correspondent aux noms des dossiers)
CLASSES = ["Parasitized", "Uninfected"]
LABELS  = {0: "Parasitized", 1: "Uninfected"}


def charger_chemins(data_dir: str) -> tuple:
    """
    Parcourt les deux dossiers du dataset et retourne les chemins
    de toutes les images avec leurs étiquettes.

    Paramètres
    ----------
    data_dir : str
        Chemin vers le dossier cell_images/
        (doit contenir Parasitized/ et Uninfected/)

    Retourne
    --------
    chemins : list[str]   – chemins complets des images
    labels  : list[int]   – 0 = Parasitized, 1 = Uninfected
    """
    chemins, labels = [], []

    for label_id, classe in enumerate(CLASSES):
        dossier = os.path.join(data_dir, classe)

        if not os.path.exists(dossier):
            raise FileNotFoundError(
                f"Dossier introuvable : '{dossier}'\n"
                f"→ Vérifiez que cell_images/ contient bien '{classe}/'"
            )

        fichiers = [
            f for f in os.listdir(dossier)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        for f in fichiers:
            chemins.append(os.path.join(dossier, f))
            labels.append(label_id)

        print(f"  [{classe}]  {len(fichiers)} images chargées (label={label_id})")

    print(f"\n[✔] Total : {len(chemins)} images  |  "
          f"Parasitized={labels.count(0)}  Uninfected={labels.count(1)}")
    return chemins, labels


def afficher_exemples(chemins: list, labels: list,
                      n: int = 8, save_path: str = "metrics/exemples_images.png") -> None:
    """
    Affiche n exemples d'images (moitié infectées, moitié saines)
    avec leurs étiquettes et sauvegarde la figure.

    Paramètres
    ----------
    chemins   : list  – chemins des images
    labels    : list  – étiquettes correspondantes
    n         : int   – nombre d'images à afficher (doit être pair)
    save_path : str   – chemin de sauvegarde de la figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # On prend n/2 images de chaque classe
    idx_parasit  = [i for i, l in enumerate(labels) if l == 0]
    idx_uninfect = [i for i, l in enumerate(labels) if l == 1]

    selectionnes  = random.sample(idx_parasit,  n // 2)
    selectionnes += random.sample(idx_uninfect, n // 2)
    random.shuffle(selectionnes)

    cols = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.5, 6))
    fig.suptitle("Exemples d'images – Détection du Paludisme",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, ax in zip(selectionnes, axes.flatten()):
        img = Image.open(chemins[idx]).resize((100, 100))
        ax.imshow(img)
        label_txt = LABELS[labels[idx]]
        couleur   = "#ef4444" if labels[idx] == 0 else "#10b981"
        ax.set_title(label_txt, fontsize=9, color=couleur, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✔] Exemples sauvegardés : {save_path}")


def stats_dataset(chemins: list, labels: list) -> None:
    """
    Affiche des statistiques simples sur le dataset :
    nombre d'images par classe, taille moyenne des images.
    """
    print("\n── Statistiques du dataset ──────────────────────────")
    print(f"  Total images    : {len(chemins)}")
    print(f"  Parasitized     : {labels.count(0)}")
    print(f"  Uninfected      : {labels.count(1)}")

    # Taille de quelques images (échantillon de 20)
    echantillon = random.sample(chemins, min(20, len(chemins)))
    tailles = [Image.open(c).size for c in echantillon]
    largeurs = [t[0] for t in tailles]
    hauteurs = [t[1] for t in tailles]
    print(f"  Taille moyenne  : {int(np.mean(largeurs))} × {int(np.mean(hauteurs))} px")
    print(f"  Taille min/max  : {min(largeurs)}×{min(hauteurs)} / {max(largeurs)}×{max(hauteurs)} px")
    print("─────────────────────────────────────────────────────\n")
