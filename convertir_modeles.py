# Lancez ce script dans votre terminal local
# python convertir_modeles.py

import tensorflow as tf
import os

models_dir = "models"  # adaptez le chemin si nécessaire

for nom in ["CNN_Simple", "CNN_Deep", "CNN_BN", "best_model"]:
    chemin_keras = os.path.join(models_dir, f"{nom}.keras")
    chemin_poids = os.path.join(models_dir, f"{nom}.weights.h5")

    if os.path.exists(chemin_keras):
        print(f"Conversion : {nom}.keras → {nom}.weights.h5")
        modele = tf.keras.models.load_model(chemin_keras)
        modele.save_weights(chemin_poids)
        print(f"[OK] {nom}.weights.h5 sauvegarde")