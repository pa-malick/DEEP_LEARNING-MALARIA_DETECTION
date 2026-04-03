# ================================================================
# main.py  –  Pipeline complète Deep Learning : Détection du Paludisme
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# Usage :
#   python main.py
#   python main.py --data data/cell_images --epochs 30
# ================================================================

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader  import charger_chemins, afficher_exemples, stats_dataset
from src.preprocessing import split_donnees, creer_generateurs
from ImageDataGenerator.models       import get_modeles
from ImageDataGenerator.train        import entrainer_tous
from ImageDataGenerator.evaluation   import (evaluer_modele, tracer_courbes,
                           comparer_modeles, sauvegarder_metriques)
from ImageDataGenerator.utils        import sauvegarder_meilleur


DATA_DIR_DEFAUT = "data/cell_images"
EPOCHS_DEFAUT   = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline DL – Détection du Paludisme"
    )
    parser.add_argument("--data",   type=str, default=DATA_DIR_DEFAUT,
                        help="Chemin vers cell_images/")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAUT,
                        help="Nombre max d'époques (défaut: 20)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("   DEEP LEARNING – MALARIA DETECTION")
    print("   Papa Malick NDIAYE | Master DSGL | UADB")
    print("=" * 60)

    # ── Étape 1 : Chargement ────────────────────────────────────
    print("\n[ÉTAPE 1 / 6]  Chargement des données")
    chemins, labels = charger_chemins(args.data)
    stats_dataset(chemins, labels)
    afficher_exemples(chemins, labels, n=8)

    # ── Étape 2 : Prétraitement ─────────────────────────────────
    print("\n[ÉTAPE 2 / 6]  Prétraitement & générateurs")
    df_train, df_val, df_test = split_donnees(chemins, labels)
    gen_train, gen_val, gen_test = creer_generateurs(df_train, df_val, df_test)

    # ── Étape 3 : Modèles ───────────────────────────────────────
    print("\n[ÉTAPE 3 / 6]  Création des modèles CNN")
    modeles = get_modeles()
    for nom, m in modeles.items():
        m.summary()

    # ── Étape 4 : Entraînement ──────────────────────────────────
    print("\n[ÉTAPE 4 / 6]  Entraînement")
    histories = entrainer_tous(modeles, gen_train, gen_val, epochs=args.epochs)

    # ── Étape 5 : Évaluation ────────────────────────────────────
    print("\n[ÉTAPE 5 / 6]  Évaluation sur les données de test")
    resultats = {}
    for nom, modele in modeles.items():
        # Courbes d'apprentissage
        tracer_courbes(nom, histories[nom])
        # Métriques sur le test
        resultats[nom] = evaluer_modele(nom, modele, gen_test)

    # ── Étape 6 : Comparaison & sauvegarde ─────────────────────
    print("\n[ÉTAPE 6 / 6]  Comparaison & sauvegarde")
    meilleur_nom = comparer_modeles(resultats)
    sauvegarder_metriques(resultats, meilleur_nom)
    sauvegarder_meilleur(modeles[meilleur_nom], meilleur_nom)

    print("\n" + "=" * 60)
    print(f"  ✅  Pipeline terminée avec succès !")
    print(f"  🏆  Meilleur modèle  : {meilleur_nom}")
    print(f"  📊  Métriques        : metrics/results.json")
    print(f"  📈  Graphiques       : metrics/")
    print(f"  💾  Modèle           : models/best_model.keras")
    print("=" * 60)
    print("\n  → Lancez maintenant l'API :  python app/app.py\n")


if __name__ == "__main__":
    main()
