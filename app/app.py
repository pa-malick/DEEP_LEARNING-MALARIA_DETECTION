# app.py – API Flask + interface web de prédiction
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
#
# Endpoints :
#   GET  /         → interface HTML
#   POST /predict  → reçoit une image, retourne la prédiction JSON
#   GET  /metrics  → métriques JSON des modèles
#   GET  /health   → statut de l'API

import sys
import os
import uuid

from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import charger_meilleur, charger_metriques, predire_image

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EXTENSIONS_VALIDES = {"png", "jpg", "jpeg", "bmp"}

BASE_DIR  = os.path.dirname(__file__)
modele    = charger_meilleur(os.path.join(BASE_DIR, "..", "models"))
metriques = charger_metriques(os.path.join(BASE_DIR, "..", "metrics", "results.json"))


def extension_valide(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in EXTENSIONS_VALIDES


@app.route("/")
def index():
    return render_template("index.html", metriques=metriques)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue."}), 400

    fichier = request.files["image"]

    if fichier.filename == "":
        return jsonify({"erreur": "Fichier vide."}), 400

    if not extension_valide(fichier.filename):
        return jsonify({"erreur": "Format non supporté. Utilisez PNG, JPG ou BMP."}), 400

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
        if os.path.exists(chemin_tmp):
            os.remove(chemin_tmp)
        return jsonify({"erreur": str(e)}), 500


@app.route("/metrics")
def get_metrics():
    return jsonify(metriques)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "API operationnelle"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[OK] API Flask demarree sur http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
