# Makefile - MALARIA_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB

.PHONY: install run run-patient serve rapport test clean docker-build docker-up docker-down docker-logs help

help:
	@echo ""
	@echo "  Commandes disponibles :"
	@echo "  make install      - Installer les dependances"
	@echo "  make run          - Pipeline DL, split par image"
	@echo "  make run-patient  - Pipeline DL, split par patient (scores honnetes)"
	@echo "  make serve        - Demarrer l'API Flask"
	@echo "  make rapport      - Generer le rapport Word"
	@echo "  make test         - Lancer les tests"
	@echo "  make clean        - Supprimer les fichiers generes"
	@echo "  make docker-up    - Lancer avec Docker"
	@echo ""

install:
	pip install -r requirements.txt

run:
	python main.py

run-patient:
	python main.py --split patient

serve:
	python app/app.py

rapport:
	python generer_rapport.py

test:
	pytest tests/ -v

clean:
	rm -f models/*.keras
	rm -f metrics/*.json metrics/*.png
	rm -f app/static/uploads/*
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	find . -name "*.pyc" -delete

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down --rmi all --volumes
