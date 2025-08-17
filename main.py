from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json, re
from model import predict_synthesis
import database as db
import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict_molecule(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.RingCount(mol)
    ]
    pred = model.predict([features])[0]
    return int(pred)
app = FastAPI(
    title="AromaAI Platform",
    version="0.1.0",
    openapi_url="/api-spec.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

SPEC_DIR = "specs"
SPEC_FILE = os.path.join(SPEC_DIR, "aromaai.json")

@app.on_event("startup")
def startup():
    os.makedirs(SPEC_DIR, exist_ok=True)
    db.init_db()
    # Сохраняем OpenAPI в JSON
    with open(SPEC_FILE, "w", encoding="utf-8") as f:
        json.dump(app.openapi(), f, ensure_ascii=False, indent=2)

@app.get("/download-spec")
def download_spec():
    return FileResponse(SPEC_FILE, filename="aromaai.json")

# --- Модели Pydantic ---

class PredictRequest(BaseModel):
    molecule: str = Field(..., example="C6H6")

class PredictResponse(BaseModel):
    input: str
    predicted_path: str
    confidence: float

class HistoryItem(BaseModel):
    id: str
    molecule: str
    predicted_path: str
    confidence: float
    timestamp: str

class HistoryResponse(BaseModel):
    predictions: List[HistoryItem]

# --- Валидация молекулы ---
SMILES_PATTERN = re.compile(r'^[A-Za-z0-9@+\-=\#\/\\\(\)\[\]\%\.\:]+$')
def is_valid_molecule(s: str) -> bool:
    return bool(s and SMILES_PATTERN.fullmatch(s))

# --- Эндпоинты ---

@app.get("/", tags=["misc"])
def read_root():
    return {"message": "AromaAI backend работает"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    predicted_label = predict_molecule(data.molecule)
    if predicted_label is None:
        raise HTTPException(status_code=422, detail="Некорректная молекула")

    # Возвращаем объект PredictResponse
    return PredictResponse(
        input=data.molecule,
        predicted_label=predicted_label,
        confidence=None  # можно позже добавить вероятность/score
    )

    path, confidence = predict_synthesis(data.molecule)
    db.add_history(data.molecule, path, confidence)
    return PredictResponse(input=data.molecule, predicted_path=path, confidence=confidence)

@app.get("/history", response_model=HistoryResponse, tags=["history"])
def get_history(limit: Optional[int] = Query(None, ge=1, le=100)):
    records = db.get_history(limit)
    return HistoryResponse(predictions=records)

@app.delete("/history", tags=["history"])
def clear_history():
    db.clear_history()
    return {"status": "ok", "message": "История очищена"}
