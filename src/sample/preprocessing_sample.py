"""
src/preprocessing.py

Production preprocessing pipeline.

Назначение:
- загрузка и агрегация raw-данных;
- применение Feature Engineering;
- объединение с целевой переменной;
- сохранение итогового датасета.

Ноутбуки используются только для исследований.
"""

from pathlib import Path
import pandas as pd

from src.sample.data_loading_sample import build_base_dataset
from src.sample.feature_engineering_sample import (
    add_overdue_structure_features,
    add_payment_risk_features,
    add_overdue_features,
    add_limit_ratio_features,
    add_payment_discipline_features,
    add_credit_timeline_features,
)

# ============================================================
# Пути
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_PATH = PROJECT_ROOT / "data" / "target" / "train_target.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Загрузка и агрегация raw-данных
# ============================================================
def load_base_dataset() -> pd.DataFrame:
    """
    Загружает parquet-файлы и агрегирует данные по id.
    """
    df = build_base_dataset()
    return df


# ============================================================
# 2. Feature Engineering
# ============================================================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет финальный набор признаков (A + B).
    """
    df = df.copy()
    df = add_overdue_features(df)
    df = add_overdue_structure_features(df)
    df = add_limit_ratio_features(df)
    df = add_payment_discipline_features(df)
    df = add_payment_risk_features(df)
    df = add_credit_timeline_features(df)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет OHE для категориальных колонок.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return df
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True)


def align_to_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Выравнивает набор фичей по сохранённому списку колонок.
    """
    return df.reindex(columns=feature_columns, fill_value=0)


# ============================================================
# 3. Объединение с таргетом
# ============================================================
def merge_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет признаки с целевой переменной.
    """
    target = pd.read_csv(TARGET_PATH)

    df_merged = df.merge(
        target,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    return df_merged


# ============================================================
# 4. Сохранение
# ============================================================
def save_dataset(df: pd.DataFrame, name: str) -> Path:
    """
    Сохраняет датасет в data/processed/.
    """
    path = PROCESSED_DIR / name
    df.to_pickle(path)
    return path


# ============================================================
# 5. Главный entrypoint
# ============================================================
def run_preprocessing(with_target: bool = True) -> Path:
    """
    Полный preprocessing pipeline.

    with_target=True:
        → dataset_with_target.pkl
    with_target=False:
        → dataset_features.pkl
    """
    print("▶ Loading raw data...")
    df = load_base_dataset()

    print("▶ Applying feature engineering...")
    # никаких обращений к enc_paym_0 здесь нет и быть не должно
    df = apply_feature_engineering(df)

    # print("▶ Aggregating FE features by id...")
    # df = aggregate_fe_by_id(df)

    assert df.index.is_unique
    assert "id" in df.columns

    if with_target:
        print("▶ Merging with target...")
        df = merge_with_target(df)
        output_name = "dataset_with_target_sample.pkl"
    else:
        output_name = "dataset_features_sample.pkl"

    path = save_dataset(df, output_name)

    print(f"✔ Dataset saved to {path}")
    return path


def aggregate_fe_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует ТОЛЬКО FE-признаки.
    enc_paym_* считаем уже агрегированными — не трогаем.
    """

    # enc_paym_* — уже *_max, их НЕ агрегируем
    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]

    # OHE по credit_type — тоже не агрегируем
    ohe_cols = [
        c for c in df.columns
        if c.startswith("enc_loans_credit_type_")
    ]

    fe_cols = [
        c for c in df.columns
        if c not in ["id", "rn"]
        and c not in paym_cols
        and c not in ohe_cols
    ]

    df_num = (
        df.groupby("id")[fe_cols]
        .agg(["mean", "max"])
        .reset_index()
    )

    df_num.columns = [
        "id" if col[0] == "id" else f"{col[0]}_{col[1]}"
        for col in df_num.columns
    ]

    # enc_paym_* + OHE → просто берём
    df_static = (
        df[["id"] + paym_cols + ohe_cols]
        .drop_duplicates("id")
        .reset_index(drop=True)
    )

    return df_num.merge(df_static, on="id", how="left")


