# чтение parquet порциями

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import numpy as np


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_parquet_files(data_dir: Path = RAW_DATA_DIR) -> List[Path]:
    """
    Возвращает список parquet-файлов с данными.
    """
    files = sorted(data_dir.glob("*.pq"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files


def load_parquet_iter(
    files: Iterable[Path],
    columns: List[str] | None = None,
) -> Iterable[pd.DataFrame]:
    """
    Итеративно читает parquet-файлы.
    """
    for file in files:
        yield pd.read_parquet(file, columns=columns)


def aggregate_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегация числовых признаков по id.
    Псевдокатегориальные признаки ИСКЛЮЧЕНЫ.
    """

    # псевдокатегориальные признаки — НЕ агрегируем как числа
    categorical_like = {
        "enc_loans_credit_type",
        "enc_loans_credit_status",
        "enc_loans_account_holder_type",
        "enc_loans_account_cur",
    }

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in categorical_like and c not in {"id"}
    ]

    agg_map = {col: ["mean", "max"] for col in numeric_cols}

    df_agg = (
        df.groupby("id", as_index=False)
          .agg(agg_map)
    )

    # выравниваем имена колонок
    df_agg.columns = [
        "id" if col[0] == "id" else f"{col[0]}_{col[1]}"
        for col in df_agg.columns
    ]

    return df_agg



def build_base_dataset() -> pd.DataFrame:
    parquet_files = get_parquet_files()

    num_chunks = []
    ohe_chunks = []

    for chunk in load_parquet_iter(parquet_files):
        # числовая агрегация (как было)
        agg_num = aggregate_by_id(chunk)
        num_chunks.append(agg_num)

        # OHE по credit_type
        agg_ohe = aggregate_ohe_share(
            chunk,
            col="enc_loans_credit_type",
            id_col="id"
        )
        ohe_chunks.append(agg_ohe)

    # объединяем чанки
    num_df = pd.concat(num_chunks, axis=0)
    num_df = num_df.groupby("id", as_index=False).mean()

    ohe_df = pd.concat(ohe_chunks, axis=0)

    # финальный merge
    dataset = num_df.merge(ohe_df, on="id", how="left")

    return dataset


def aggregate_ohe_share(df: pd.DataFrame, col: str, id_col: str = "id") -> pd.DataFrame:
    """
    One-Hot Encode категориального признака и агрегация долей по id
    """
    ohe = pd.get_dummies(df[col], prefix=col)
    ohe[id_col] = df[id_col]

    return (
        ohe
        .groupby(id_col, as_index=False)
        .mean()
    )



