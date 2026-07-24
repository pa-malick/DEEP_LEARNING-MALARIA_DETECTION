# Dockerfile - MALARIA_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL, UADB

FROM python:3.13-slim

LABEL maintainer="Papa Malick NDIAYE <njaymika@gmail.com>"
LABEL description="Malaria Detection - Deep Learning CNN + Flask API"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

# On copie requirements.txt en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models metrics app/static/uploads

# L'application ne doit pas tourner en root dans le conteneur
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# gunicorn et non le serveur de developpement Flask.
# 1 worker : le modele Keras occupe environ 100 Mo en memoire, et le plan
# gratuit de Render est limite a 512 Mo. Les threads suffisent pour la charge.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --pythonpath app --workers 1 --threads 4 --timeout 120 app:app"]
