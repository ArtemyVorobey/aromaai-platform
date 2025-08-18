from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Словарь с доступными моделями
MODELS = {
    "chemberta": "seyonec/PubChem10M_SMILES_BERT",  # химия SMILES
    "toxicity": "molecule/toxicity-pubchem"         # пример модели для токсичности
}

# Выбираем модель по умолчанию
DEFAULT_MODEL = "chemberta"

class MoleculeModel:
    def __init__(self, model_name=DEFAULT_MODEL):
        if model_name not in MODELS:
            raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(MODELS.keys())}")
        self.model_id = MODELS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)

    def predict(self, smiles: str):
        """
        Предсказание свойства молекулы по SMILES.
        Возвращает словарь: {"predicted_class": int, "confidence": float}
        """
        try:
            inputs = self.tokenizer(smiles, return_tensors="pt")
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            return {"predicted_class": predicted_class, "confidence": confidence}
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None


# Инициализация глобального объекта модели
current_model = MoleculeModel(DEFAULT_MODEL)

def predict_molecule(smiles: str):
    return current_model.predict(smiles)
