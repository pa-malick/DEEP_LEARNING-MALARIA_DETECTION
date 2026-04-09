<div align="center">

# 🦟 DEEP LEARNING – MALARIA DETECTION

### Détection automatique du paludisme par analyse d'images de cellules sanguines

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Papa Malick NDIAYE** · Master Data Science & Génie Logiciel · Université Alioune Diop de Bambey

🐙 **GitHub :** https://github.com/pa-malick/MALARIA_DETECTION
💼 **LinkedIn :** https://www.linkedin.com/in/papa-malick-ndiaye-b58b22309
🖥️ **Démo_live :** https://deep-learning-malaria-detection-znf2.onrender.com

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
- ✅ Interface web futuriste (Three.js + Anime.js + GSAP)
- ✅ Containerisé avec Docker
- ✅ Tests unitaires pytest

---

## 📊 Résultats obtenus

| Modèle | Accuracy | Précision | Rappel | F1-Score |
|--------|----------|-----------|--------|----------|
| CNN_Simple | 96.23% | 95.48% | 97.05% | 96.26% |
| **CNN_Deep** | **96.64%** | **96.30%** | **97.00%** | **96.65%** |
| CNN_BN | 95.86% | 94.59% | 97.29% | 95.92% |

🏆 **Meilleur modèle : CNN_Deep** avec **96.64% d'accuracy**

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
│   ├── app.py                ← API Flask (4 endpoints)
│   ├── templates/index.html  ← Interface futuriste (Three.js + GSAP)
│   └── static/css + js/
├── models/                   ← Modèles .keras (générés)
├── metrics/                  ← Graphiques & métriques (générés)
├── tests/                    ← Tests unitaires pytest
├── main.py                   ← Pipeline complète
├── Dockerfile                ← Image Docker
├── docker-compose.yml        ← Orchestration Docker
├── requirements.txt
└── Makefile
```

---

## 🚀 Installation & Utilisation

### Option 1 — Local (Python)

```bash
# 1. Cloner
git clone https://github.com/pa-malick/MALARIA_DETECTION.git
cd MALARIA_DETECTION

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Placer le dataset
# → data/cell_images/Parasitized/ et data/cell_images/Uninfected/

# 4. Lancer la pipeline ML
python main.py

# 5. Démarrer l'API Flask
python app/app.py
# → http://localhost:5000
```

### Option 2 — Docker

```bash
# Construire et lancer en une commande
docker-compose up --build

# → http://localhost:5000
```

---

## 🧠 Architectures CNN

| Modèle | Blocs Conv | Spécificité |
|--------|-----------|-------------|
| CNN_Simple | 2 | Baseline rapide |
| CNN_Deep | 3 | Plus de capacité |
| CNN_BN | 3 + BatchNorm | Entraînement stable |

---

## 🖥️ Interface Web

Interface futuriste avec :
- **Three.js** — fond de particules 3D animé
- **GSAP** — animations d'entrée et scroll
- **Anime.js** — micro-animations UI
- **Drag & Drop** — upload d'image intuitif
- **Jauge de confiance** — probabilité animée

---

## 🛠️ Technologies

| Catégorie | Outils |
|-----------|--------|
| Deep Learning | TensorFlow / Keras |
| Traitement image | PIL, ImageDataGenerator |
| Visualisation | Matplotlib, Seaborn |
| API | Flask |
| Interface | Three.js, GSAP, Anime.js |
| Déploiement | Docker, Render |
| Tests | Pytest |

---

## 💡 Améliorations possibles

- [ ] Transfer Learning avec MobileNetV2 (>97% accuracy)
- [ ] Grad-CAM pour visualiser les zones d'attention du modèle
- [ ] Déploiement sur Render ou Hugging Face Spaces
- [ ] Application mobile (TensorFlow Lite)

---

## 📥 Dataset

**NIH Malaria Dataset** — 27 558 images PNG :
- [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- [NIH Official](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)

---

## 👨‍💻 Auteur

**Papa Malick NDIAYE**
Master Data Science & Génie Logiciel — Université Alioune Diop de Bambey (UADB)
📧 njaymika@gmail.com
🐙 [github.com/pa-malick/MALARIA_DETECTION](https://github.com/pa-malick/MALARIA_DETECTION)
💼 [linkedin.com/in/papa-malick-ndiaye-b58b22309](https://www.linkedin.com/in/papa-malick-ndiaye-b58b22309)
🖥️ (https://deep-learning-malaria-detection-znf2.onrender.com)

---

<div align="center">
<sub>Projet réalisé dans le cadre du Master DSGL – UADB © 2025</sub>
</div>
