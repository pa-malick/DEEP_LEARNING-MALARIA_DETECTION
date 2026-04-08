# data_loader.py – Chargement et visualisation du dataset
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


CLASSES = ["Parasitized", "Uninfected"]
LABELS  = {0: "Parasitized", 1: "Uninfected"}


def charger_chemins(data_dir: str) -> tuple:
    """Récupère les chemins de toutes les images et leurs étiquettes."""
    chemins = []
    labels  = []

    for label_id, classe in enumerate(CLASSES):
        dossier = os.path.join(data_dir, classe)

        if not os.path.exists(dossier):
            raise FileNotFoundError(
                f"Dossier introuvable : '{dossier}'\n"
                f"Vérifiez que cell_images/ contient bien '{classe}/'"
            )

        fichiers = [
            f for f in os.listdir(dossier)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        for f in fichiers:
            chemins.append(os.path.join(dossier, f))
            labels.append(label_id)

        print(f"  [{classe}]  {len(fichiers)} images chargées")

    print(f"\n  Total : {len(chemins)} images  |  "
          f"Parasitized={labels.count(0)}  Uninfected={labels.count(1)}")

    return chemins, labels


def afficher_exemples(chemins: list, labels: list,
                      n: int = 8,
                      save_path: str = "metrics/exemples_images.png") -> None:
    """Sauvegarde une grille de n exemples d'images (moitié infectées, moitié saines)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    print(f"  Exemples sauvegardés : {save_path}")


def stats_dataset(chemins: list, labels: list) -> None:
    """Affiche quelques statistiques de base sur le dataset."""
    print("\n── Statistiques du dataset ──────────────────────────")
    print(f"  Total images    : {len(chemins)}")
    print(f"  Parasitized     : {labels.count(0)}")
    print(f"  Uninfected      : {labels.count(1)}")

    # Vérification sur un échantillon pour ne pas parcourir les 27 000 images
    echantillon = random.sample(chemins, min(20, len(chemins)))
    tailles  = [Image.open(c).size for c in echantillon]
    largeurs = [t[0] for t in tailles]
    hauteurs = [t[1] for t in tailles]

    print(f"  Taille moyenne  : {int(np.mean(largeurs))} × {int(np.mean(hauteurs))} px")
    print(f"  Taille min/max  : {min(largeurs)}×{min(hauteurs)} / {max(largeurs)}×{max(hauteurs)} px")
    print("─────────────────────────────────────────────────────\n")
