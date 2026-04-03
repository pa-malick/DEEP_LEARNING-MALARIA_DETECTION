# ================================================================
# app.py  –  API Flask + interface de prédiction
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# Endpoints :
#   GET  /           → interface web
#   POST /predict    → prédiction sur image uploadée
#   GET  /metrics    → métriques JSON
#   GET  /health     → statut API
# ================================================================

import sys
import os
import uuid
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ImageDataGenerator.utils import charger_meilleur, charger_metriques, predire_image

app = Flask(__name__)

# Dossier temporaire pour les images uploadées
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Formats d'images acceptés
EXTENSIONS_VALIDES = {"png", "jpg", "jpeg", "bmp"}

# Chargement du modèle au démarrage
BASE_DIR = os.path.dirname(__file__)
modele   = charger_meilleur(os.path.join(BASE_DIR, "..", "models"))
metriques = charger_metriques(os.path.join(BASE_DIR, "..", "metrics", "results.json"))


def extension_valide(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in EXTENSIONS_VALIDES


@app.route("/")
def index():
    return render_template("index.html", metriques=metriques)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit une image de cellule sanguine et retourne la prédiction.
    L'image est sauvegardée temporairement, analysée, puis supprimée.
    """
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue."}), 400

    fichier = request.files["image"]

    if fichier.filename == "":
        return jsonify({"erreur": "Fichier vide."}), 400

    if not extension_valide(fichier.filename):
        return jsonify({"erreur": "Format non supporté. Utilisez PNG, JPG ou BMP."}), 400

    # Sauvegarde temporaire
    ext        = fichier.filename.rsplit(".", 1)[1].lower()
    nom_temp   = f"{uuid.uuid4().hex}.{ext}"
    chemin_tmp = os.path.join(UPLOAD_DIR, nom_temp)
    fichier.save(chemin_tmp)

    try:
        resultat = predire_image(modele, chemin_tmp)
        return jsonify({
            "label"      : resultat["label"],
            "probabilite": resultat["probabilite"],
            "classe_id"  : resultat["classe_id"],
            "image_url"  : f"/static/uploads/{nom_temp}"
        })
    except Exception as e:
        os.remove(chemin_tmp)
        return jsonify({"erreur": str(e)}), 500


@app.route("/metrics")
def get_metrics():
    return jsonify(metriques)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "API opérationnelle ✔"})


if __name__ == "__main__":
    print("\n🚀  API Flask démarrée sur  http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
