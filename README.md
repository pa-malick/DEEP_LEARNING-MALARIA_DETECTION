# Détection du paludisme par deep learning

Classification d'images de cellules sanguines (parasitées / saines) par réseaux
de neurones convolutifs, avec une API Flask de démonstration.

**Papa Malick NDIAYE** · Master Data Science & Génie Logiciel · Université Alioune Diop de Bambey

- Démo : https://deep-learning-malaria-detection-znf2.onrender.com
- Code : https://github.com/pa-malick/DEEP_LEARNING-MALARIA_DETECTION
- Contact : njaymika@gmail.com

> Projet académique. Cet outil n'est pas un dispositif médical et ne constitue
> pas un diagnostic. Voir la section [Limites](#limites).

## Contexte

Le diagnostic du paludisme par microscopie est lent et dépend de l'expertise de
l'opérateur. Ce projet évalue dans quelle mesure trois architectures CNN de
complexité croissante peuvent classer automatiquement des cellules sanguines
issues du jeu de données NIH.

## Résultats

Découpage **par patient** : les 200 patients sont répartis en 139 / 31 / 30,
aucun n'apparaît dans deux ensembles. Le jeu de test compte 3 451 images
issues de 30 patients jamais vus à l'entraînement. La classe positive est
`Parasitized`.

| Modèle | Accuracy | Sensibilité | Spécificité | F1 |
|---|---|---|---|---|
| CNN_Simple | 96,78 % | 95,44 % | 97,71 % | 96,02 % |
| CNN_Deep (retenu) | 96,78 % | 94,65 % | 98,24 % | 95,99 % |
| CNN_BN | 96,12 % | 94,51 % | 97,22 % | 95,19 % |

Le modèle retenu est sélectionné sur le **jeu de validation**, jamais sur le
test. Le test ne sert qu'à l'estimation finale.

### Un seul chiffre ne suffit pas

Pour le modèle retenu, les scores diffèrent nettement selon le sous-ensemble
de patients évalué, alors qu'il s'agit du même modèle et du même protocole :

| CNN_Deep | Validation (31 patients) | Test (30 patients) |
|---|---|---|
| Accuracy | 93,94 % | 96,78 % |
| Sensibilité | 89,46 % | 94,65 % |

5,2 points d'écart en sensibilité. Avec seulement une trentaine de patients
par ensemble, l'estimation est instable : le jeu de test est tombé sur des
patients plus faciles que le jeu de validation. La lecture honnête est donc
que **la sensibilité du modèle se situe entre 89 % et 95 % selon les
patients**, et non qu'elle vaut 94,65 %.

Cette dispersion était invisible avec un découpage par image, où validation
et test donnaient 96,03 % et 96,01 %. Cette concordance était un artefact de
la fuite entre patients, pas une preuve de robustesse.

Deux métriques comptent ici plus que l'accuracy :

- **Sensibilité** : proportion de cellules parasitées effectivement détectées.
  C'est la métrique critique en dépistage, car un faux négatif est un malade
  renvoyé chez lui. Le modèle retenu manque 75 cellules parasitées sur 1 403.
- **Spécificité** : proportion de cellules saines correctement classées.

## Limites

Ces limites sont connues et non corrigées à ce stade. Elles doivent être lues
avant toute interprétation des chiffres ci-dessus.

1. **L'estimation repose sur un seul découpage, et elle est instable.** Comme
   détaillé plus haut, la sensibilité passe de 89,5 % à 94,7 % selon que l'on
   évalue sur les 31 patients de validation ou les 30 patients de test. Trente
   patients, c'est trop peu pour un chiffre stable. Une validation croisée par
   groupes (`GroupKFold`, 5 plis) donnerait une moyenne et un écart-type,
   c'est-à-dire le seul résultat réellement défendable. Elle n'a pas été faite,
   son coût étant d'environ cinq fois celui d'un entraînement complet.

2. **Les trois architectures sont équivalentes.** CNN_Simple et CNN_Deep
   obtiennent la même accuracy (96,78 %) et CNN_BN est à 0,7 point. Chaque
   architecture n'a été entraînée qu'une fois. Vu la dispersion mesurée au
   point 1, ces écarts sont du bruit. La conclusion utile est que la
   profondeur supplémentaire n'apporte rien ici, et qu'à performance égale
   CNN_Simple est préférable puisqu'il est le plus léger.

3. **Environ 5 % des cellules parasitées ne sont pas détectées** sur le jeu de
   test (75 sur 1 403), et jusqu'à 10 % sur le jeu de validation. C'est
   incompatible avec un usage clinique.

4. **Validation limitée au jeu NIH** : frottis minces, coloration Giemsa,
   images redimensionnées en 64 x 64. Rien ne garantit la transposition à
   d'autres protocoles de préparation ou d'autres microscopes.

5. **Aucune validation clinique, aucun avis médical.** Le projet est un exercice
   d'apprentissage automatique, pas un outil de santé.

## Structure

```
DEEP_LEARNING-MALARIA_DETECTION/
├── data/cell_images/      Dataset (Parasitized/ + Uninfected/), non versionné
├── src/
│   ├── data_loader.py     Chargement et statistiques
│   ├── preprocessing.py   Split, normalisation, générateurs
│   ├── models.py          3 architectures CNN
│   ├── train.py           Entraînement et callbacks
│   ├── evaluation.py      Métriques, courbes, sélection
│   └── utils.py           Sauvegarde, chargement, prédiction
├── app/
│   ├── app.py             API Flask
│   ├── templates/         Interface web
│   └── static/            CSS et JS
├── models/                Modèles .keras générés, non versionnés
├── metrics/               Graphiques et métriques générés
├── tests/                 Tests pytest
├── main.py                Pipeline complète
├── generer_rapport.py     Génération du rapport Word
├── Dockerfile
└── requirements.txt
```

## Installation

Le dataset et les modèles ne sont pas versionnés (400 Mo et 100 Mo). Il faut
donc télécharger le premier et régénérer les seconds.

```bash
git clone https://github.com/pa-malick/DEEP_LEARNING-MALARIA_DETECTION.git
cd DEEP_LEARNING-MALARIA_DETECTION
pip install -r requirements.txt
```

Placer ensuite le dataset dans `data/cell_images/Parasitized/` et
`data/cell_images/Uninfected/`, puis lancer la pipeline :

```bash
python main.py --split patient    # protocole des résultats publiés, environ 1 h sur CPU
python main.py                    # découpage par image, conservé à titre de comparaison
python app/app.py                 # API sur http://localhost:5000
```

Avec Docker :

```bash
docker compose up --build
```

Tests (45 tests, ne nécessitent ni dataset ni modèle entraîné) :

```bash
pytest tests/ -v
```

## Rapport

Le rapport Word n'est pas écrit à la main. Il est reconstruit depuis
`metrics/results.json` et `metrics/histories.json`, de sorte que ses chiffres
ne peuvent pas diverger de ceux produits par la pipeline.

```bash
python generer_rapport.py
```

## Architectures

| Modèle | Blocs conv | Particularité |
|---|---|---|
| CNN_Simple | 2 | Baseline |
| CNN_Deep | 3 | Plus de capacité |
| CNN_BN | 3 | Batch Normalization |

Entrée 64 x 64 x 3, sortie sigmoïde, `binary_crossentropy`, optimiseur Adam.
Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
Augmentation (rotation, translation, flip, zoom) appliquée au train uniquement.

## Technologies

TensorFlow / Keras, scikit-learn, Pillow, Matplotlib, Seaborn, Flask, gunicorn,
Docker, pytest.

## Pistes d'amélioration

- Validation croisée `GroupKFold` par patient, pour une moyenne et un écart-type
- Plusieurs runs par architecture, avec des graines différentes
- Ajustement du seuil de décision pour privilégier la sensibilité
- Transfer learning (MobileNetV2)
- Grad-CAM pour visualiser les zones d'attention du modèle

## Dataset

NIH Malaria Dataset, 27 558 images PNG équilibrées entre les deux classes.

- https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- https://ceb.nlm.nih.gov/repositories/malaria-datasets/

## Licence

MIT, voir [LICENSE](LICENSE).
