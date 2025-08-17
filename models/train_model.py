import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- Пример: генерация фич из SMILES ---
def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*5  # заглушка для некорректных молекул
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.RingCount(mol)
    ]

# --- Загружаем данные ---
# Пример: CSV с колонками 'smiles' и 'label' (0/1)
df = pd.read_csv("data/molecules.csv")

# Генерация признаков
X = df['smiles'].apply(smiles_to_features).tolist()
y = df['label'].tolist()

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Обучение модели ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Проверка точности
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Сохраняем модель ---
joblib.dump(model, "model.pkl")
print("Модель сохранена в model.pkl")
