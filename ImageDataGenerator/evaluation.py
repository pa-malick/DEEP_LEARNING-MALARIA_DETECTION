# ================================================================
# evaluation.py  –  Évaluation, visualisations et comparaison
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

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


# ── 1. Évaluation d'un modèle ────────────────────────────────────

def evaluer_modele(nom: str, modele, gen_test) -> dict:
    """
    Évalue un modèle sur le générateur de test et calcule
    accuracy, précision, rappel et F1-score.

    Retourne
    --------
    dict  –  métriques du modèle
    """
    # Prédictions sur tout le set de test
    gen_test.reset()
    y_proba = modele.predict(gen_test, verbose=0)
    y_pred  = (y_proba > 0.5).astype(int).flatten()
    y_true  = gen_test.classes

    metriques = {
        "accuracy"  : round(float(accuracy_score(y_true, y_pred)), 4),
        "precision" : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"    : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score"  : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    print(f"\n  ── {nom} ──")
    print(f"     Accuracy  : {metriques['accuracy']  * 100:.2f} %")
    print(f"     Précision : {metriques['precision'] * 100:.2f} %")
    print(f"     Rappel    : {metriques['recall']    * 100:.2f} %")
    print(f"     F1-Score  : {metriques['f1_score']  * 100:.2f} %")
    print("\n  Rapport de classification :")
    print(classification_report(y_true, y_pred,
                                target_names=["Parasitized", "Uninfected"],
                                zero_division=0))

    _tracer_matrice_confusion(nom, y_true, y_pred)
    return metriques


# ── 2. Matrice de confusion ──────────────────────────────────────

def _tracer_matrice_confusion(nom: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=["Parasitized", "Uninfected"],
        yticklabels=["Parasitized", "Uninfected"],
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Matrice de confusion – {nom}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Prédit", fontsize=9)
    ax.set_ylabel("Réel",   fontsize=9)

    os.makedirs("metrics", exist_ok=True)
    chemin = f"metrics/cm_{nom}.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"    → Matrice sauvegardée : {chemin}")


# ── 3. Courbes d'apprentissage ───────────────────────────────────

def tracer_courbes(nom: str, history: dict) -> None:
    """
    Trace et sauvegarde les courbes loss et accuracy
    pour l'entraînement et la validation.

    Ces courbes permettent de visualiser :
      - Si le modèle apprend bien (loss décroissante)
      - S'il y a surapprentissage (val_loss remonte alors que train_loss baisse)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Courbes d'apprentissage – {nom}",
                 fontsize=12, fontweight="bold")

    epochs_range = range(1, len(history["loss"]) + 1)

    # Courbe de loss
    axes[0].plot(epochs_range, history["loss"],     label="Train loss",  color="#3b82f6", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Val loss",    color="#ef4444", linewidth=2, linestyle="--")
    axes[0].set_title("Loss", fontsize=10)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Courbe d'accuracy
    axes[1].plot(epochs_range, history["accuracy"],     label="Train acc", color="#10b981", linewidth=2)
    axes[1].plot(epochs_range, history["val_accuracy"], label="Val acc",   color="#f59e0b", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy", fontsize=10)
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    os.makedirs("metrics", exist_ok=True)
    chemin = f"metrics/learning_curves_{nom}.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"[✔] Courbes sauvegardées : {chemin}")


# ── 4. Comparaison des modèles ───────────────────────────────────

def comparer_modeles(resultats: dict) -> str:
    """
    Affiche un tableau comparatif et génère un graphique.
    Sélectionne le meilleur modèle selon l'accuracy.

    Retourne
    --------
    str  –  nom du meilleur modèle
    """
    print("\n┌─ COMPARAISON DES MODÈLES ─────────────────────────────┐")
    print(f"  {'Modèle':<15} {'Accuracy':>9} {'Précision':>10} {'Rappel':>8} {'F1':>8}")
    print("  " + "─" * 48)

    for nom, m in resultats.items():
        print(
            f"  {nom:<15}"
            f"  {m['accuracy']  * 100:>7.2f}%"
            f"  {m['precision'] * 100:>8.2f}%"
            f"  {m['recall']    * 100:>6.2f}%"
            f"  {m['f1_score']  * 100:>6.2f}%"
        )

    meilleur = max(resultats, key=lambda k: resultats[k]["accuracy"])
    print(f"\n  🏆  Meilleur modèle (accuracy) : {meilleur}  "
          f"→  {resultats[meilleur]['accuracy'] * 100:.2f}%")
    print("└───────────────────────────────────────────────────────┘\n")

    _tracer_comparaison(resultats, meilleur)
    return meilleur


def _tracer_comparaison(resultats: dict, meilleur: str) -> None:
    noms = list(resultats.keys())
    acc  = [resultats[n]["accuracy"]  * 100 for n in noms]
    prec = [resultats[n]["precision"] * 100 for n in noms]
    rec  = [resultats[n]["recall"]    * 100 for n in noms]
    f1   = [resultats[n]["f1_score"]  * 100 for n in noms]

    x = np.arange(len(noms))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - 1.5*w, acc,  w, label="Accuracy",  color="#3b82f6", edgecolor="white")
    b2 = ax.bar(x - 0.5*w, prec, w, label="Précision", color="#10b981", edgecolor="white")
    b3 = ax.bar(x + 0.5*w, rec,  w, label="Rappel",    color="#f59e0b", edgecolor="white")
    b4 = ax.bar(x + 1.5*w, f1,   w, label="F1-Score",  color="#ef4444", edgecolor="white")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(noms, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("Comparaison des 3 CNN – Détection du Paludisme",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    idx = noms.index(meilleur)
    ax.axvspan(idx - 0.48, idx + 0.48, alpha=0.07, color="#3b82f6")

    plt.tight_layout()
    plt.savefig("metrics/comparaison_modeles.png", dpi=150)
    plt.close()
    print("[✔] Comparaison sauvegardée : metrics/comparaison_modeles.png")


# ── 5. Sauvegarde JSON ───────────────────────────────────────────

def sauvegarder_metriques(resultats: dict, meilleur: str) -> None:
    os.makedirs("metrics", exist_ok=True)
    payload = {"meilleur_modele": meilleur, "resultats": resultats}
    with open("metrics/results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print("[✔] Métriques exportées : metrics/results.json")
