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
    Используем mean/max для числовых и моду для категориальных.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != id_col]

    agg_numeric = (
        df.groupby(id_col, as_index=False)[numeric_cols]
        .agg(["mean", "max"])
    )
    agg_numeric.columns = [
        f"{col}_{stat}" if stat else col
        for col, stat in agg_numeric.columns
    ]

    if cat_cols:
        agg_cat = (
            df.groupby(id_col, as_index=False)[cat_cols]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        )
        agg_df = agg_numeric.merge(agg_cat, on=id_col, how="left")
    else:
        agg_df = agg_numeric

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

    dataset = pd.concat(aggregated_chunks, axis=0)
    numeric_cols = dataset.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]
    cat_cols = dataset.select_dtypes(include=["object", "category"]).columns.tolist()

    agg_map: dict[str, str | callable] = {col: "mean" for col in numeric_cols}
    for col in cat_cols:
        agg_map[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]

    dataset = dataset.groupby("id", as_index=False).agg(agg_map)
    return dataset
