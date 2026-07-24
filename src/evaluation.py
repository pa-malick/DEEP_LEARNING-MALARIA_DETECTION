# evaluation.py - Évaluation, visualisations et comparaison des modèles
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB
#
# Convention de classes (imposée par flow_from_dataframe, ordre alphabétique) :
#   Parasitized = 0   (classe positive au sens clinique : cellule malade)
#   Uninfected  = 1
#
# Les métriques cliniques sont donc calculées avec pos_label=0.
#   sensibilite = rappel sur Parasitized  = proportion de malades détectés
#   specificite = rappel sur Uninfected   = proportion de sains non alarmés
#   precision   = valeur predictive positive sur Parasitized

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


PARASITIZED = 0
UNINFECTED  = 1


def calculer_metriques(y_true, y_pred) -> dict:
    """Calcule les métriques en prenant Parasitized comme classe positive."""
    return {
        "accuracy"   : round(float(accuracy_score(y_true, y_pred)), 4),
        "sensibilite": round(float(recall_score(y_true, y_pred,
                                                pos_label=PARASITIZED, zero_division=0)), 4),
        "specificite": round(float(recall_score(y_true, y_pred,
                                                pos_label=UNINFECTED, zero_division=0)), 4),
        "precision"  : round(float(precision_score(y_true, y_pred,
                                                   pos_label=PARASITIZED, zero_division=0)), 4),
        "f1_score"   : round(float(f1_score(y_true, y_pred,
                                            pos_label=PARASITIZED, zero_division=0)), 4),
    }


def predire(modele, generateur) -> tuple:
    """Retourne (y_true, y_pred) pour un générateur non mélangé."""
    generateur.reset()
    y_proba = modele.predict(generateur, verbose=0)
    y_pred  = (y_proba > 0.5).astype(int).flatten()
    return generateur.classes, y_pred


def evaluer_modele(nom: str, modele, gen_test, tracer: bool = True) -> dict:
    """Évalue un modèle sur le jeu de test et affiche le détail par classe."""
    y_true, y_pred = predire(modele, gen_test)
    metriques = calculer_metriques(y_true, y_pred)

    print(f"\n  == {nom} ==")
    print(f"     Accuracy    : {metriques['accuracy']    * 100:.2f} %")
    print(f"     Sensibilite : {metriques['sensibilite'] * 100:.2f} %   (malades detectes)")
    print(f"     Specificite : {metriques['specificite'] * 100:.2f} %   (sains non alarmes)")
    print(f"     Precision   : {metriques['precision']   * 100:.2f} %   (VPP sur Parasitized)")
    print(f"     F1-Score    : {metriques['f1_score']    * 100:.2f} %")

    cm = confusion_matrix(y_true, y_pred, labels=[PARASITIZED, UNINFECTED])
    faux_negatifs = int(cm[0, 1])
    total_malades = int(cm[0].sum())
    print(f"\n     Malades non detectes : {faux_negatifs} / {total_malades}")

    print("\n  Rapport de classification :")
    print(classification_report(y_true, y_pred,
                                target_names=["Parasitized", "Uninfected"],
                                zero_division=0))

    if tracer:
        _tracer_matrice_confusion(nom, cm)
    return metriques


def selectionner_meilleur(modeles: dict, gen_val) -> tuple:
    """
    Choisit le meilleur modèle sur le jeu de VALIDATION.

    La sélection ne doit jamais utiliser le jeu de test : celui-ci sert
    uniquement à estimer la performance finale du modèle retenu.
    """
    print("\n  Selection sur le jeu de validation :")
    scores = {}

    for nom, modele in modeles.items():
        y_true, y_pred = predire(modele, gen_val)
        m = calculer_metriques(y_true, y_pred)
        scores[nom] = m
        print(f"    {nom:<12}  accuracy {m['accuracy'] * 100:6.2f} %"
              f"   sensibilite {m['sensibilite'] * 100:6.2f} %")

    meilleur = max(scores, key=lambda k: scores[k]["accuracy"])
    print(f"\n  Modele retenu : {meilleur} (accuracy validation "
          f"{scores[meilleur]['accuracy'] * 100:.2f} %)")
    return meilleur, scores


def _tracer_matrice_confusion(nom: str, cm) -> None:
    """Génère et sauvegarde la matrice de confusion."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Parasitized", "Uninfected"],
        yticklabels=["Parasitized", "Uninfected"],
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(f"Matrice de confusion - {nom}", fontsize=10)
    ax.set_xlabel("Predit", fontsize=9)
    ax.set_ylabel("Reel", fontsize=9)

    os.makedirs("metrics", exist_ok=True)
    chemin = f"metrics/cm_{nom}.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"    Matrice sauvegardee : {chemin}")


def tracer_courbes(nom: str, history: dict) -> None:
    """Trace et sauvegarde les courbes loss et accuracy par époque."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Courbes d'apprentissage - {nom}", fontsize=12)

    epochs_range = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs_range, history["loss"],     label="Train loss", linewidth=1.5)
    axes[0].plot(epochs_range, history["val_loss"], label="Val loss",   linewidth=1.5, linestyle="--")
    axes[0].set_title("Loss", fontsize=10)
    axes[0].set_xlabel("Epoque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history["accuracy"],     label="Train acc", linewidth=1.5)
    axes[1].plot(epochs_range, history["val_accuracy"], label="Val acc",   linewidth=1.5, linestyle="--")
    axes[1].set_title("Accuracy", fontsize=10)
    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    os.makedirs("metrics", exist_ok=True)
    chemin = f"metrics/learning_curves_{nom}.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"  Courbes sauvegardees : {chemin}")


def comparer_modeles(resultats: dict, meilleur: str) -> None:
    """Affiche le tableau comparatif des performances sur le jeu de test."""
    print("\n  Performances sur le jeu de test :")
    print(f"    {'Modele':<13}{'Accuracy':>10}{'Sensib.':>10}{'Specif.':>10}{'F1':>9}")
    print("    " + "-" * 52)

    for nom, m in resultats.items():
        marque = " *" if nom == meilleur else "  "
        print(
            f"    {nom:<13}"
            f"{m['accuracy']    * 100:>9.2f}%"
            f"{m['sensibilite'] * 100:>9.2f}%"
            f"{m['specificite'] * 100:>9.2f}%"
            f"{m['f1_score']    * 100:>8.2f}%{marque}"
        )

    print("\n    * modele retenu (selection faite sur la validation)")
    _tracer_comparaison(resultats, meilleur)


def _tracer_comparaison(resultats: dict, meilleur: str) -> None:
    """Graphique en barres groupées comparant les modèles sur le test."""
    noms = list(resultats.keys())
    series = [
        ("Accuracy",    [resultats[n]["accuracy"]    * 100 for n in noms]),
        ("Sensibilite", [resultats[n]["sensibilite"] * 100 for n in noms]),
        ("Specificite", [resultats[n]["specificite"] * 100 for n in noms]),
        ("F1-Score",    [resultats[n]["f1_score"]    * 100 for n in noms]),
    ]

    x = np.arange(len(noms))
    w = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, valeurs) in enumerate(series):
        offset = (i - 1.5) * w
        barres = ax.bar(x + offset, valeurs, w, label=label)
        for barre in barres:
            h = barre.get_height()
            ax.text(barre.get_x() + barre.get_width() / 2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(noms, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title(f"Comparaison des 3 CNN sur le jeu de test (retenu : {meilleur})",
                 fontsize=12, pad=12)
    ax.legend(fontsize=9, ncol=4, loc="lower center")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    os.makedirs("metrics", exist_ok=True)
    plt.tight_layout()
    plt.savefig("metrics/comparaison_modeles.png", dpi=150)
    plt.close()
    print("  Comparaison sauvegardee : metrics/comparaison_modeles.png")


def sauvegarder_metriques(resultats: dict, meilleur: str,
                          scores_val: dict = None) -> None:
    """Exporte les métriques en JSON pour l'API Flask."""
    os.makedirs("metrics", exist_ok=True)
    payload = {
        "meilleur_modele"  : meilleur,
        "critere_selection": "accuracy sur le jeu de validation",
        "classe_positive"  : "Parasitized",
        "resultats"        : resultats,
    }
    if scores_val:
        payload["validation"] = scores_val

    with open("metrics/results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print("  Metriques exportees : metrics/results.json")


def sauvegarder_historiques(histories: dict) -> None:
    """Sauvegarde les historiques d'entraînement pour pouvoir retracer les courbes."""
    os.makedirs("metrics", exist_ok=True)
    serialisable = {
        nom: {cle: [float(v) for v in valeurs] for cle, valeurs in hist.items()}
        for nom, hist in histories.items()
    }
    with open("metrics/histories.json", "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=4)
    print("  Historiques exportes : metrics/histories.json")
