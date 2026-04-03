# ================================================================
# Makefile – DEEP_LEARNING-MALARIA_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
# ================================================================

.PHONY: install run serve test clean help

help:
	@echo ""
	@echo "  MALARIA_DETECTION – Commandes disponibles"
	@echo "  ──────────────────────────────────────────"
	@echo "  make install   →  Installer les dépendances"
	@echo "  make run       →  Lancer la pipeline DL complète"
	@echo "  make serve     →  Démarrer l'API Flask (port 5000)"
	@echo "  make test      →  Lancer les tests unitaires"
	@echo "  make clean     →  Supprimer les fichiers générés"
	@echo ""

install:
	pip install -r requirements.txt

run:
	python main.py

serve:
	python app/app.py

test:
	pytest tests/ -v --tb=short

clean:
	rm -f models/*.keras models/*.h5
	rm -f metrics/*.json metrics/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Nettoyage terminé."
