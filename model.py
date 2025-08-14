# model.py
from typing import Tuple

def predict_synthesis(molecule: str) -> Tuple[str, float]:
    """
    Заглушка для реальной модели синтеза.
    Ha этом этапе можно подключить ML-модель, симулятор или API.
    
    Возвращает:
        predicted_path: str — путь синтеза
        confidence: float — уверенность модели (0.0–1.0)
    """
    # Здесь пока простой пример:
    if "C" in molecule:
        return "Синтез по стандартной схеме", 0.85
    else:
        return "Неизвестная схема синтеза", 0.5
