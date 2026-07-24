# app.py - API Flask + interface web de prédiction
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB
#
# Endpoints :
#   GET  /         -> interface HTML
#   POST /predict  -> reçoit une image, retourne la prédiction JSON
#   GET  /metrics  -> métriques JSON des modèles
#   GET  /health   -> statut de l'API

import sys
import os
import uuid

from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import charger_meilleur, charger_metriques, predire_image

app = Flask(__name__)

# Un fichier de plus de 8 Mo n'est pas une image de cellule.
# Sans cette limite, un seul upload peut saturer le disque du conteneur.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EXTENSIONS_VALIDES = {"png", "jpg", "jpeg", "bmp"}

BASE_DIR  = os.path.dirname(__file__)
metriques = charger_metriques(os.path.join(BASE_DIR, "..", "metrics", "results.json"))

# Le modèle est chargé au démarrage, mais une absence ne doit pas empêcher
# l'API de répondre : /health reste disponible pour diagnostiquer le problème.
try:
    modele = charger_meilleur(os.path.join(BASE_DIR, "..", "models"))
except Exception as erreur:
    modele = None
    print(f"[ERREUR] Modele non charge : {erreur}")


def extension_valide(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in EXTENSIONS_VALIDES


@app.route("/")
def index():
    return render_template("index.html", metriques=metriques)


@app.route("/predict", methods=["POST"])
def predict():
    # La requête est validée avant de vérifier la disponibilité du modèle :
    # une requête invalide reste une erreur 400, que le modèle soit chargé ou non.
    if "image" not in request.files:
        return jsonify({"erreur": "Aucune image reçue."}), 400

    fichier = request.files["image"]

    if fichier.filename == "":
        return jsonify({"erreur": "Fichier vide."}), 400

    if not extension_valide(fichier.filename):
        return jsonify({"erreur": "Format non supporté. Utilisez PNG, JPG ou BMP."}), 400

    if modele is None:
        return jsonify({"erreur": "Modele indisponible sur le serveur."}), 503

    ext        = fichier.filename.rsplit(".", 1)[1].lower()
    chemin_tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    fichier.save(chemin_tmp)

    try:
        resultat = predire_image(modele, chemin_tmp)
        return jsonify({
            "label"      : resultat["label"],
            "probabilite": resultat["probabilite"],
            "proba_parasite": resultat["proba_parasite"],
            "classe_id"  : resultat["classe_id"],
        })
    except Exception as erreur:
        # Le détail est journalisé côté serveur, pas renvoyé au client.
        print(f"[ERREUR] Prediction echouee : {erreur}")
        return jsonify({"erreur": "Image illisible ou format invalide."}), 400
    finally:
        # L'image de l'utilisateur ne doit pas être conservée sur le serveur.
        if os.path.exists(chemin_tmp):
            os.remove(chemin_tmp)


@app.errorhandler(413)
def fichier_trop_gros(_):
    return jsonify({"erreur": "Image trop volumineuse (8 Mo maximum)."}), 413


@app.route("/metrics")
def get_metrics():
    return jsonify(metriques)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if modele is not None else "degraded",
        "modele_charge": modele is not None,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[OK] API Flask demarree sur http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
