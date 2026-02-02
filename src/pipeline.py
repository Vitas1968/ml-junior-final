"""


Production training pipeline.

Назначение:
- запуск preprocessing;
- обучение финальной модели;
- оценка качества;
- сохранение артефактов.

Ноутбуки НЕ используются.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from src.preprocessing import run_preprocessing
from src.modeling import train_model

# ============================================================
# Пути
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"


# ============================================================
# 1. Загрузка данных (через preprocessing)
# ============================================================
def load_dataset() -> pd.DataFrame:
    """
    Запускает preprocessing и возвращает датасет с таргетом.
    """
    dataset_path = run_preprocessing(with_target=True)
    df = pd.read_pickle(dataset_path)
    return df


# ============================================================
# 2. Train / Test split
# ============================================================
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["flag", "id"])
    y = df["flag"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )




# ============================================================
# 4. Оценка качества
# ============================================================
def evaluate_model(model, X_test, y_test) -> float:
    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    return roc_auc


# ============================================================
# 5. Сохранение артефактов
# ============================================================
def save_validation_predictions(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred_proba": preds,
        }
    ).to_csv(PREDICTIONS_PATH, index=False)

# ============================================================
# 6. Обучение на полном датасете
# ============================================================
def train_full_model(df: pd.DataFrame):
    X = df.drop(columns=["flag", "id"])
    y = df["flag"]

    model = train_model(X, y)
    return model
# ============================================================
# 6. Главный entrypoint
# ============================================================
def main():
    print("▶ Loading dataset...")
    df = load_dataset()

    print("▶ Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("▶ Training model (train split) ...")
    model = train_model(X_train, y_train)

    print("▶ Evaluating model...")
    roc_auc = evaluate_model(model, X_test, y_test)
    print(f"ROC-AUC (test): {roc_auc:.4f}")

    print("▶ Training FINAL model on full dataset...")
    final_model = train_full_model(df)

    print("▶ Saving artifacts...")
    save_validation_predictions(model, X_test, y_test)

    print("▶ Saving FINAL model...")
    joblib.dump(final_model, MODEL_PATH)

    print("✔ Pipeline completed successfully")


if __name__ == "__main__":
    main()