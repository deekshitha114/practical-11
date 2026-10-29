# Student Result Prediction API — Practical 11
## MLOps: Logging and Error Tracking

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/student-result-prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/student-result-prediction/actions)

---

### 📌 Project Overview

A FastAPI-based ML service that predicts whether a student will **Pass** or **Fail**
based on anonymised academic performance features. **Practical-11** adds structured
application logging and error tracking to the previously deployed container service.

### 🆕 Practical-11 Additions

| Feature | Description |
|---------|-------------|
| **Structured logging** | Consistent `timestamp \| level \| name \| message` format to stdout |
| **HTTP middleware** | Every request logged with method, path, status, duration, request_id |
| **Request ID** | UUID per request — echoed in `X-Request-ID` response header |
| **Validation handler** | 422 failures logged with field locations (no payload copying) |
| **Prediction logging** | Feature names logged; SAFE_INPUT_FIELDS controls value logging |
| **Exception tracking** | `logger.exception()` captures stack traces for 500 errors |
| **Model-load events** | Success/failure at startup visible in container logs |

---

### 🗂️ Project Structure

```
11-mlops/
├── app.py              # FastAPI service + Practical-11 logging
├── train_model.py      # Generates & saves model.pkl
├── test_app.py         # Pytest smoke tests (used by CI)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── .dockerignore
├── .gitignore
└── .github/
    └── workflows/
        └── ci-cd.yml   # GitHub Actions CI/CD pipeline
```

---

### 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and save the model
python train_model.py

# 3. Start the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 4. Open Swagger UI
# http://127.0.0.1:8000/docs
```

---

### 🐳 Docker

```bash
# Build
docker build -t student-result-prediction:practical-11 .

# Run
docker run --rm -p 8000:8000 --name srp-p11 student-result-prediction:practical-11

# Test health
curl http://localhost:8000/health
```

---

### 🔍 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Service health / readiness check |
| `POST` | `/predict` | Student Pass/Fail prediction |
| `GET`  | `/docs` | Interactive Swagger UI |
| `GET`  | `/redoc` | ReDoc documentation |

#### Example `/predict` request body
```json
{
  "attendance":      85.0,
  "study_hours":     5.0,
  "previous_scores": 78.0,
  "assignments":     9,
  "extracurricular": 1
}
```

#### Example `/predict` response
```json
{
  "prediction": "Pass",
  "request_id": "3f2a1c0e-...",
  "version": "1.0.0"
}
```

---

### 📋 Log Levels

| Level | Event | Example |
|-------|-------|---------|
| `INFO` | Model loaded, health check, valid prediction | `event=model_loaded` |
| `WARNING` | 422 validation error, 404 missing route | `event=validation_error` |
| `ERROR` | Prediction exception (500) | `event=prediction_failed` |

---

### 🌐 Deployment

The image is published to **GitHub Container Registry (GHCR)**:
```
ghcr.io/YOUR_USERNAME/student-result-prediction:latest
```

The service is deployed on **Render** using the GHCR image.  
See [Practical-10 README] for Render setup instructions.

---

### ✅ CI/CD Pipeline

```
git push → GitHub Actions
  ├── Job 1: pytest smoke tests
  ├── Job 2: Docker build + health/predict/422 checks
  └── Job 3: Push to GHCR (main branch only)
              ↓
         Render manual redeploy
```

---

### ⚠️ Privacy Notice

- Student PII (name, ID, email) is **never** passed to or logged by this API.
- `SAFE_INPUT_FIELDS` in `app.py` controls which feature values (if any) appear in logs.
- By default, only feature **names** are logged — never values.
