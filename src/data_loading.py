# чтение parquet порциями

from pathlib import Path
from typing import Iterable, List

import pandas as pd


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
        df = pd.read_parquet(file, columns=columns)
        yield df


def aggregate_by_id(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    Базовая агрегация признаков на уровне заявки (id).
    Используем mean как безопасный baseline.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col]

    agg_df = (
        df.groupby(id_col, as_index=False)[numeric_cols]
        .mean()
    )
    return agg_df


def build_base_dataset() -> pd.DataFrame:
    """
    Основная функция:
    - читает parquet-файлы итеративно
    - агрегирует признаки по id
    - объединяет всё в единый DataFrame
    """
    parquet_files = get_parquet_files()

    aggregated_chunks = []

    for chunk in load_parquet_iter(parquet_files):
        agg_chunk = aggregate_by_id(chunk)
        aggregated_chunks.append(agg_chunk)

    dataset = (
        pd.concat(aggregated_chunks, axis=0)
        .groupby("id", as_index=False)
        .mean()
    )

    return dataset
