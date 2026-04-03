<div align="center">

# 🦟 DEEP LEARNING – MALARIA DETECTION

### Détection automatique du paludisme par analyse d'images de cellules sanguines

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?style=flat-square&logo=bootstrap)](https://getbootstrap.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Papa Malick NDIAYE** · Master Data Science & Génie Logiciel · Université Alioune Diop de Bambey

</div>

---

## 📌 Contexte & Problématique

Le paludisme reste l'une des maladies infectieuses les plus meurtrières au monde, touchant des millions de personnes chaque année, particulièrement en Afrique subsaharienne. Le diagnostic traditionnel par microscopie est **lent**, **coûteux** et **dépend de l'expertise humaine**.

Ce projet propose une solution d'IA pour **automatiser la détection du parasite *Plasmodium*** dans les images de frottis sanguins, en comparant trois architectures CNN de complexité croissante.

> *"Chaque seconde compte dans le diagnostic du paludisme."*

---

## ✨ Fonctionnalités

- ✅ Chargement automatique du dataset cell_images (Parasitized / Uninfected)
- ✅ Affichage des exemples d'images avec étiquettes colorées
- ✅ Augmentation des données (rotation, flip, zoom) pour éviter l'overfitting
- ✅ 3 architectures CNN comparées : Simple, Deep, BatchNorm
- ✅ Courbes d'apprentissage (loss + accuracy) par modèle
- ✅ Matrices de confusion + métriques complètes
- ✅ API Flask avec upload d'image et prédiction en temps réel
- ✅ Interface web Bootstrap responsive avec drag & drop
- ✅ Tests unitaires pytest

---

## 🗂️ Structure du projet

```
MALARIA_DETECTION/
├── data/cell_images/         ← Dataset (Parasitized/ + Uninfected/)
├── src/
│   ├── data_loader.py        ← Chargement & visualisation
│   ├── preprocessing.py      ← Split, normalisation, ImageDataGenerator
│   ├── models.py             ← 3 architectures CNN
│   ├── train.py              ← Entraînement avec callbacks
│   ├── evaluation.py         ← Métriques, courbes, comparaison
│   └── utils.py              ← Save/load modèle, prédiction unitaire
├── app/
│   ├── app.py                ← API Flask
│   ├── templates/index.html  ← Interface web
│   └── static/css + js/
├── models/                   ← Modèles sauvegardés (générés)
├── metrics/                  ← Graphiques & métriques (générés)
├── tests/                    ← Tests unitaires pytest
├── main.py                   ← Pipeline complète
├── requirements.txt
└── Makefile
```

---

## 🚀 Installation & Utilisation

```bash
# 1. Cloner
git clone https://github.com/votre-username/MALARIA_DETECTION.git
cd MALARIA_DETECTION

# 2. Installer les dépendances
make install

# 3. Placer le dataset
# Copier cell_images/ dans data/
# Structure attendue :
#   data/cell_images/Parasitized/  (images .png)
#   data/cell_images/Uninfected/   (images .png)

# 4. Lancer la pipeline
make run

# 5. Démarrer l'API Flask
make serve
# → http://localhost:5000

# 6. Tests
make test
```

---

## 🧠 Architectures CNN

| Modèle | Blocs Conv | Paramètres | Spécificité |
|--------|-----------|------------|-------------|
| CNN_Simple | 2 | ~200K | Baseline rapide |
| CNN_Deep   | 3 | ~500K | Plus de capacité |
| CNN_BN     | 3 + BN | ~500K | Entraînement plus stable |

---

## 📊 Résultats attendus

Avec le dataset NIH Cell Images (27 558 images) et 20 époques d'entraînement :

| Modèle | Accuracy attendue |
|--------|------------------|
| CNN_Simple | ~93% |
| CNN_Deep   | ~94% |
| CNN_BN     | ~95% |

---

## 🖥️ Interface Web

L'interface permet de glisser-déposer une image de cellule sanguine et d'obtenir instantanément la prédiction du modèle avec un score de confiance.

---

## 🛠️ Technologies

| Catégorie | Outils |
|-----------|--------|
| Deep Learning | TensorFlow / Keras |
| Traitement image | PIL, ImageDataGenerator |
| Visualisation | Matplotlib, Seaborn |
| API | Flask |
| Interface | HTML, CSS, Bootstrap 5, JavaScript |
| Tests | Pytest |

---

## 💡 Améliorations possibles

- [ ] Transfer Learning avec MobileNetV2 ou EfficientNet (>97% accuracy)
- [ ] Grad-CAM pour visualiser ce que le modèle "regarde"
- [ ] Déploiement sur Render.com ou Hugging Face Spaces
- [ ] Application mobile (TensorFlow Lite)

---

## 📥 Dataset

Le dataset utilisé est le **NIH Malaria Dataset** disponible sur :
- [Kaggle – Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- [NIH Official](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)

---

## 👨‍💻 Auteur

**Papa Malick NDIAYE**  
Master Data Science & Génie Logiciel — Université Alioune Diop de Bambey (UADB)  
📧 njaymika@gmail.com

---

<div align="center">
<sub>Projet réalisé dans le cadre du Master DSGL – UADB © 2025</sub>
</div>
