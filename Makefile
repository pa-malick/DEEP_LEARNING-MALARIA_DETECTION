# Makefile – MALARIA_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

.PHONY: install run serve test clean docker-build docker-up docker-down docker-logs help

help:
	@echo ""
	@echo "  Commandes disponibles :"
	@echo "  make install      - Installer les dépendances"
	@echo "  make run          - Lancer la pipeline DL"
	@echo "  make serve        - Démarrer l'API Flask"
	@echo "  make test         - Lancer les tests"
	@echo "  make clean        - Supprimer les fichiers générés"
	@echo "  make docker-up    - Lancer avec Docker"
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
	rm -f models/best_model.keras
	rm -f metrics/*.json metrics/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

docker-build:
	docker-compose build

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down --rmi all --volumes
