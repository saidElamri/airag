# 🤖 Assistant RAG Support IT - Manuel d'Utilisation

Ce projet est un assistant intelligent basé sur la technique RAG (Retrieval-Augmented Generation) conçu pour aider les techniciens IT à partir de documentations PDF.

## 🚀 Fonctionnalités Clés
- **RAG Pipeline**: Recherche sémantique via ChromaDB et LLM (Groq Llama-3 / Google Gemini).
- **Backend Sécurisé**: FastAPI avec authentification JWT (PBKDF2).
- **Tracé & Monitoring**: Tracking complet des requêtes et latence avec **MLflow**.
- **Clustering**: Regroupement thématique des questions utilisateurs (KMeans).
- **DevOps**: Docker, Docker-compose et déploiement Kubernetes.
- **CI/CD**: Automatisation via GitHub Actions.

## 🛠️ Installation et Lancement

### 1. Pré-requis
- Python 3.13
- Docker & Docker Compose
- Clés API : Groq (recommandé) ou Google Gemini

### 2. Configuration (.env)
Créez un fichier `.env` à la racine :
```env
GOOGLE_API_KEY=votre_cle_google
GROQ_API_KEY=votre_cle_groq
SECRET_KEY=une_cle_secrete_aleatoire
DATABASE_URL=sqlite:///./sql_app.db
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_DIR=./chroma_db
```

### 3. Lancement Local
```bash
# Installation
pip install -r requirements.txt

# Ingestion du PDF (Premier lancement)
python ingest.py

# Démarrage de l'API
uvicorn main:app --reload
```

### 4. Lancement avec Docker Compose (PostgreSQL)
```bash
docker-compose up --build
```

## 📊 Monitoring (MLflow)
Pour visualiser les performances du RAG, les prompts et la latence :
```bash
mlflow ui --port 5000
```
Le pipeline RAG est également enregistré dans le **Model Registry** sous le nom `IT-Support-Assistant`.

## 🛰️ API Documentation
| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/auth/login` | Obtention du token JWT |
| POST | `/query` | Poser une question au RAG (Auth requis) |
| GET | `/history` | Historique des questions/réponses |
| GET | `/health` | État du service |

## 🧪 Tests & Qualité
Les tests de validité sont automatisés via **GitHub Actions** à chaque push sur `main` ou `develop`.

---
*Réalisé dans le cadre du projet Certification RNCP Développeur.se en intelligence artificielle.*
