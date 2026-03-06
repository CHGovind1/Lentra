# ML Platform - German Credit Risk Classification

A production-quality ML training pipeline and inference service for credit risk classification using the German Credit dataset.

## Overview

This project implements a complete ML platform with:
- **Data Ingestion** - Downloads data from UCI repository
- **Preprocessing** - Parses data, remaps targets, encodes categoricals
- **Feature Engineering** - Creates derived features (monthly burden, debt load, age groups)
- **Model Training** - XGBoost classifier with MLflow tracking
- **Inference Service** - FastAPI REST API for predictions

## Project Structure

```
.
├── config/
│   └── config.yaml          # All configuration parameters
├── src/
│   ├── pipeline/
│   │   ├── ingest.py        # Data download
│   │   ├── preprocess.py    # Data preprocessing
│   │   ├── features.py      # Feature engineering
│   │   ├── train.py         # Model training + MLflow
│   │   └── evaluate.py      # Model evaluation
│   └── serve/
│       └── app.py           # FastAPI inference service
├── docker-compose.yml       # All services orchestration
├── Dockerfile.train         # Training container
├── Dockerfile.serve        # Inference container
└── requirements.txt         # Python dependencies
```

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.9+ (for local development)

### Run Full Stack

```bash
# Start all services
docker-compose up
```

This will:
1. Start MinIO at http://localhost:9000 (credentials: minioadmin/minioadmin)
2. Start MLflow at http://localhost:5000
3. Run training pipeline
4. Start inference service at http://localhost:8000

### Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run data ingestion
python -m src.pipeline.ingest

# Run preprocessing
python -m src.pipeline.preprocess

# Run training
python -m src.pipeline.train

# Start API
uvicorn src.serve.app:app --reload
```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "Production"
}
```

### Predict
```bash
POST /predict
```

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [{
      "checking_status": "A11",
      "duration": 6,
      "credit_history": "A32",
      "purpose": "A43",
      "credit_amount": 1169,
      "savings_status": "A61",
      "employment": "A73",
      "installment_commitment": 4,
      "personal_status": "A93",
      "other_parties": "A101",
      "residence_since": 4,
      "property_magnitude": "A121",
      "age": 67,
      "other_payment_plans": "A143",
      "housing": "A152",
      "existing_credits": 2,
      "job": "A173",
      "num_dependents": 1,
      "own_telephone": "A192",
      "foreign_worker": "A201"
    }]
  }'
```

Example response:
```json
{
  "predictions": [{
    "predicted_class": 0,
    "class_label": "Good Credit Risk",
    "probability_good": 0.85,
    "probability_bad": 0.15
  }]
}
```

## Design Decisions

1. **Config-Driven**: All parameters (paths, URLs, hyperparameters) are externalized in `config.yaml`
2. **Modular Pipeline**: Each stage (ingest, preprocess, features, train, evaluate) is a separate module
3. **MLflow Integration**: Parameters, metrics, and model artifacts are logged to MLflow with MinIO storage
4. **Feature Engineering**: Added 3 derived features (monthly_burden, debt_load, age_group)
5. **Input Validation**: Pydantic models validate incoming requests

## Git Commits

This project follows Conventional Commits specification:
- `feat(pipeline): add feature engineering module` - New features
- `fix(serve): resolve prediction error` - Bug fixes
- `chore(compose): add health checks` - Configuration changes
- `docs(readme): add quick start guide` - Documentation

## License

This project is for educational purposes. The German Credit dataset is from UCI ML Repository (CC BY 4.0).
