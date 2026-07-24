# main.py - Pipeline complète : Détection du Paludisme par Deep Learning
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB
#
# Usage :
#   python main.py
#   python main.py --data data/cell_images --epochs 30 --seed 42

import sys
import os
import argparse
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader   import charger_chemins, afficher_exemples, stats_dataset
from preprocessing import split_donnees, creer_generateurs
from models        import get_modeles
from train         import entrainer_tous
from evaluation    import (evaluer_modele, tracer_courbes, comparer_modeles,
                            selectionner_meilleur, sauvegarder_metriques,
                            sauvegarder_historiques)
from utils         import sauvegarder_meilleur


DATA_DIR_DEFAUT = "data/cell_images"
EPOCHS_DEFAUT   = 20
SEED_DEFAUT     = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline DL - Malaria Detection")
    parser.add_argument("--data",   type=str, default=DATA_DIR_DEFAUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAUT)
    parser.add_argument("--seed",   type=int, default=SEED_DEFAUT)
    parser.add_argument(
        "--split", choices=["image", "patient"], default="image",
        help="patient : aucun patient partage entre train/val/test. "
             "Scores plus bas mais representatifs de nouveaux patients."
    )
    return parser.parse_args()


def fixer_seed(seed: int) -> None:
    """Fixe les graines aléatoires pour rendre un run reproductible."""
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    args = parse_args()
    fixer_seed(args.seed)

    print("\n" + "=" * 60)
    print("   DEEP LEARNING - MALARIA DETECTION")
    print("   Papa Malick NDIAYE | Master DSGL | UADB")
    print(f"   seed = {args.seed}")
    print("=" * 60)

    print("\n[ETAPE 1 / 6]  Chargement des données")
    chemins, labels = charger_chemins(args.data)
    stats_dataset(chemins, labels, seed=args.seed)
    afficher_exemples(chemins, labels, n=8, seed=args.seed)

    print("\n[ETAPE 2 / 6]  Prétraitement & générateurs")
    df_train, df_val, df_test = split_donnees(chemins, labels, seed=args.seed,
                                              par_patient=(args.split == "patient"))
    gen_train, gen_val, gen_test = creer_generateurs(df_train, df_val, df_test,
                                                      seed=args.seed)

    print("\n[ETAPE 3 / 6]  Création des modèles CNN")
    modeles = get_modeles()
    for nom, m in modeles.items():
        m.summary()

    print("\n[ETAPE 4 / 6]  Entraînement")
    histories = entrainer_tous(modeles, gen_train, gen_val, epochs=args.epochs)
    sauvegarder_historiques(histories)
    for nom in modeles:
        tracer_courbes(nom, histories[nom])

    print("\n[ETAPE 5 / 6]  Sélection du meilleur modèle (validation)")
    meilleur_nom, scores_val = selectionner_meilleur(modeles, gen_val)

    print("\n[ETAPE 6 / 6]  Évaluation finale sur le jeu de test")
    resultats = {nom: evaluer_modele(nom, modele, gen_test)
                 for nom, modele in modeles.items()}

    comparer_modeles(resultats, meilleur_nom)
    sauvegarder_metriques(resultats, meilleur_nom, scores_val)
    sauvegarder_meilleur(modeles[meilleur_nom], meilleur_nom)

    print(f"\nPipeline terminée. Modèle retenu : {meilleur_nom}")
    print("Lancez maintenant : python app/app.py\n")


if __name__ == "__main__":
    main()
