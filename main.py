from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import re
from datetime import datetime
import uuid
from fastapi.responses import FileResponse
import json
import os

# --- Настройка приложения ---
app = FastAPI(
    title="AromaAI Platform",
    version="0.1.0",
    openapi_url="/api-spec.json",  # путь к OpenAPI в браузере
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Константы для OpenAPI файла ---
SPEC_DIR = "specs"
SPEC_FILE = os.path.join(SPEC_DIR, "aromaai.json")


@app.on_event("startup")
def save_openapi_spec():
    """Сохраняем OpenAPI JSON в кастомный файл при старте сервера"""
    os.makedirs(SPEC_DIR, exist_ok=True)
    with open(SPEC_FILE, "w", encoding="utf-8") as f:
        json.dump(app.openapi(), f, ensure_ascii=False, indent=2)


@app.get("/download-spec", tags=["meta"])
def download_spec():
    """Отдаём файл OpenAPI для скачивания"""
    return FileResponse(SPEC_FILE, filename="aromaai.json")


# --- "База данных" в памяти ---
history: List[dict] = []

# --- Модели ---
class PredictRequest(BaseModel):
    molecule: str = Field(..., example="C6H6")

class PredictResponse(BaseModel):
    input: str
    predicted_path: str
    confidence: float

class ModelInfo(BaseModel):
    name: str
    description: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

class HistoryItem(BaseModel):
    id: str
    molecule: str
    predicted_path: str
    confidence: float
    timestamp: str

class HistoryResponse(BaseModel):
    predictions: List[HistoryItem]

class ValidateRequest(BaseModel):
    molecule: str

class ValidateResponse(BaseModel):
    molecule: str
    valid: bool
    message: str


# --- Утилиты ---
SMILES_PATTERN = re.compile(r'^[A-Za-z0-9@+\-=\#\/\\\(\)\[\]\%\.\:]+$')

def is_valid_molecule(s: str) -> bool:
    """Простая проверка формата молекулы (SMILES-like)."""
    if not s or not isinstance(s, str):
        return False
    return bool(SMILES_PATTERN.fullmatch(s))


# --- Эндпоинты ---
@app.get("/", tags=["misc"])
def read_root():
    return {"message": "AromaAI backend работает"}


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(data: PredictRequest):
    if not is_valid_molecule(data.molecule):
        raise HTTPException(status_code=422, detail="Недопустимый формат молекулы (символы)")

    predicted_path = "Синтез по стандартной схеме"
    confidence = 0.85

    response = PredictResponse(
        input=data.molecule,
        predicted_path=predicted_path,
        confidence=confidence
    )

    history_record = {
        "id": str(uuid.uuid4()),
        "molecule": data.molecule,
        "predicted_path": predicted_path,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    history.append(history_record)

    return response


@app.get("/models", response_model=ModelsResponse, tags=["meta"])
def get_models():
    return ModelsResponse(
        models=[
            ModelInfo(name="StandardSynth", description="Базовая модель синтеза (правила + эвристики)"),
            ModelInfo(name="AdvancedSynth", description="Модель с графовыми/генеративными компонентами")
        ]
    )


@app.get("/history", response_model=HistoryResponse, tags=["history"])
def get_history(limit: Optional[int] = Query(None, ge=1, le=100, description="Ограничить количество возвращаемых записей")):
    items = history[-limit:] if limit else history[:]
    payload = [HistoryItem(**item) for item in items]
    return HistoryResponse(predictions=payload)


@app.delete("/history", tags=["history"])
def clear_history():
    history.clear()
    return {"status": "ok", "message": "История очищена"}


@app.post("/validate", response_model=ValidateResponse, tags=["validate"])
def validate_molecule(data: ValidateRequest):
    valid = is_valid_molecule(data.molecule)
    return ValidateResponse(
        molecule=data.molecule,
        valid=valid,
        message="Формат молекулы корректный" if valid else "Недопустимые символы в молекуле"
    )

