# main.py – Pipeline complète : Détection du Paludisme par Deep Learning
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
#
# Usage :
#   python main.py
#   python main.py --data data/cell_images --epochs 30

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader   import charger_chemins, afficher_exemples, stats_dataset
from preprocessing import split_donnees, creer_generateurs
from models        import get_modeles
from train         import entrainer_tous
from evaluation    import (evaluer_modele, tracer_courbes,
                            comparer_modeles, sauvegarder_metriques)
from utils         import sauvegarder_meilleur


DATA_DIR_DEFAUT = "data/cell_images"
EPOCHS_DEFAUT   = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline DL – Malaria Detection")
    parser.add_argument("--data",   type=str, default=DATA_DIR_DEFAUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAUT)
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("   DEEP LEARNING – MALARIA DETECTION")
    print("   Papa Malick NDIAYE | Master DSGL | UADB")
    print("=" * 60)

    print("\n[ETAPE 1 / 6]  Chargement des données")
    chemins, labels = charger_chemins(args.data)
    stats_dataset(chemins, labels)
    afficher_exemples(chemins, labels, n=8)

    print("\n[ETAPE 2 / 6]  Prétraitement & générateurs")
    df_train, df_val, df_test = split_donnees(chemins, labels)
    gen_train, gen_val, gen_test = creer_generateurs(df_train, df_val, df_test)

    print("\n[ETAPE 3 / 6]  Création des modèles CNN")
    modeles = get_modeles()
    for nom, m in modeles.items():
        m.summary()

    print("\n[ETAPE 4 / 6]  Entraînement")
    histories = entrainer_tous(modeles, gen_train, gen_val, epochs=args.epochs)

    print("\n[ETAPE 5 / 6]  Évaluation")
    resultats = {}
    for nom, modele in modeles.items():
        tracer_courbes(nom, histories[nom])
        resultats[nom] = evaluer_modele(nom, modele, gen_test)

    print("\n[ETAPE 6 / 6]  Comparaison & sauvegarde")
    meilleur_nom = comparer_modeles(resultats)
    sauvegarder_metriques(resultats, meilleur_nom)
    sauvegarder_meilleur(modeles[meilleur_nom], meilleur_nom)

    print(f"\nPipeline terminée ! Meilleur modèle : {meilleur_nom}")
    print("Lancez maintenant : python app/app.py\n")


if __name__ == "__main__":
    main()
