# Dockerfile – MALARIA_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB

FROM python:3.10-slim

LABEL maintainer="Papa Malick NDIAYE <njaymika@gmail.com>"
LABEL description="Malaria Detection – Deep Learning CNN + Flask API"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

# On copie requirements.txt en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models metrics data

EXPOSE 5000

CMD ["python", "app/app.py"]
