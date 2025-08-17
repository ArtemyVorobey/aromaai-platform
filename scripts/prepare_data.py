# scripts/prepare_data.py

from rdkit import Chem

def main():
    print("Script started")

    # Проверка импорта RDKit
    try:
        mol = Chem.MolFromSmiles('CC')
        if mol:
            print("RDKit OK: молекула создана из SMILES 'CC'")
        else:
            print("RDKit FAIL: не удалось создать молекулу")
    except Exception as e:
        print("Ошибка при использовании RDKit:", e)

    print("Script finished")

if __name__ == "__main__":
    main()
