from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "AromaAI backend работает"}

class PredictRequest(BaseModel):
    molecule: str

@app.post("/predict")
def predict(data: PredictRequest):
    return {
        "input": data.molecule,
        "predicted_path": "Синтез по стандартной схеме",
        "confidence": 0.85
    }
