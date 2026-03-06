"""Inference Service - FastAPI application for predictions"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow.pyfunc
import yaml
import os
from pathlib import Path


app = FastAPI(
    title="German Credit Risk Prediction API",
    description="ML inference service for credit risk classification",
    version="1.0.0"
)

# Global model variable
model = None
model_version = None


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        current = Path(__file__).resolve()
        project_root = current.parent.parent
        config_path = project_root / config_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model():
    """Load model from MLflow registry."""
    global model, model_version
    
    config = load_config()
    model_name = config['mlflow']['model_name']
    model_stage = config['api']['model_stage']
    
    # MLflow tracking URI from environment or config
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    mlflow.set_tracking_uri(tracking_uri)
    
    # Load model from registry
    model_uri = f"models:/{model_name}/{model_stage}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    model_version = model_stage
    
    print(f"Model loaded successfully: {model_name} ({model_stage})")
    return model


# Pydantic models for request/response
class CreditApplication(BaseModel):
    checking_status: str = Field(..., description="Status of existing checking account")
    duration: int = Field(..., description="Loan duration in months")
    credit_history: str = Field(..., description="History of past credits")
    purpose: str = Field(..., description="Purpose of the loan")
    credit_amount: float = Field(..., description="Loan amount in DM")
    savings_status: str = Field(..., description="Savings account balance")
    employment: str = Field(..., description="Years of present employment")
    installment_commitment: float = Field(..., description="Installment rate as % of disposable income")
    personal_status: str = Field(..., description="Personal status and sex")
    other_parties: str = Field(..., description="Other debtors or guarantors")
    residence_since: int = Field(..., description="Years at present residence")
    property_magnitude: str = Field(..., description="Most valuable available property")
    age: int = Field(..., description="Age in years")
    other_payment_plans: str = Field(..., description="Other installment plans")
    housing: str = Field(..., description="Housing situation")
    existing_credits: int = Field(..., description="Number of existing credits")
    job: str = Field(..., description="Job type")
    num_dependents: int = Field(..., description="Number of dependents")
    own_telephone: str = Field(..., description="Telephone registration")
    foreign_worker: str = Field(..., description="Foreign worker status")


class PredictionRequest(BaseModel):
    applications: List[CreditApplication]


class PredictionResponse(BaseModel):
    predictions: List[dict]


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict credit risk for one or more applications."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame for prediction
        data = [app.dict() for app in request.applications]
        predictions = []
        
        for record in data:
            # Make prediction
            pred = model.predict([record])[0]
            prob = model.predict_proba([record])[0]
            
            predictions.append({
                "predicted_class": int(pred),
                "class_label": "Bad Credit Risk" if pred == 1 else "Good Credit Risk",
                "probability_good": float(prob[0]),
                "probability_bad": float(prob[1])
            })
        
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])
