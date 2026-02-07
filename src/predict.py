"""
Production inference script.

Назначение:
- загрузка обученной модели;
- подготовка признаков через preprocessing;
- инференс по конкретному пользователю (id);
- интерпретация вероятности дефолта.

processed-файлы напрямую НЕ используются.
"""

from pathlib import Path
import random
import joblib
import json
import pandas as pd

from src.preprocessing import (
    load_base_dataset,
    apply_feature_engineering,
    encode_categoricals,
    align_to_feature_columns,
)

# ============================================================
# Пути
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"


# ============================================================
# Интерпретация риска
# ============================================================
def interpret_risk(proba: float) -> str:
    p = proba * 100

    if p < 20:
        return "низкий риск"
    elif p < 50:
        return "умеренный риск"
    elif p < 70:
        return "высокий риск"
    else:
        return "критический риск"


# ============================================================
# Предсказание для одного пользователя
# ============================================================
def predict_for_user(model, df_features: pd.DataFrame, user_id: int) -> float:
    user_df = df_features[df_features["id"] == user_id]

    if user_df.empty:
        raise ValueError(f"id={user_id} не найден")

    proba = model.predict_proba(user_df.drop(columns=["id"]))[:, 1][0]
    return float(proba)


# ============================================================
# CLI
# ============================================================
def main():
    # 1. Проверка наличия модели
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
           "Файл model.pkl не найден. "
           "Сначала запусти: python -m src.pipeline"
        )
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
           "Файл feature_columns.json не найден. "
           "Сначала запусти: python -m src.pipeline"
        )
    # 2. Проверка наличия raw-данных
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists() or not any(raw_dir.glob("*.pq")):
        raise FileNotFoundError(
           "Raw-данные не найдены в data/raw/. "
           "Проверьте, что parquet-файлы train_data_*.pq присутствуют."
        )
    print("▶ Loading model...")
    model = joblib.load(MODEL_PATH)

    print("▶ Preparing features via preprocessing...")
    df = load_base_dataset()
    df = apply_feature_engineering(df)
    df_encoded = encode_categoricals(df)
    feature_columns = json.loads(
        FEATURE_COLUMNS_PATH.read_text(encoding="utf-8")
    )
    features = align_to_feature_columns(
        df_encoded.drop(columns=["id"]),
        feature_columns,
    )
    df_features = pd.concat([df["id"], features], axis=1)

    # --------------------------------------------------------
    # Подсказка: 10 случайных id
    # --------------------------------------------------------
    print("\nСлучайные id из датасета:")
    sample_ids = random.sample(df_features["id"].tolist(), 10)
    for uid in sample_ids:
        print(uid)

    # --------------------------------------------------------
    # Ручной ввод
    # --------------------------------------------------------
    print("\nВведите id пользователя (Enter — выход):")

    while True:
        raw = input("id = ").strip()
        if raw == "":
            break

        try:
            uid = int(raw)
            proba = predict_for_user(model, df_features, uid)
            risk = interpret_risk(proba)

            print(
                f"Вероятность дефолта пользователя с id={uid} "
                f"примерно {proba * 100:.2f} % и имеет {risk}"
            )

        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
