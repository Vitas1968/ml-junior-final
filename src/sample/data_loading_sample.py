# src/sample/data_loading_sample.py

from pathlib import Path
from typing import Iterable, List

import pandas as pd


# ============================================================
# Пути
# ============================================================
RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


# ============================================================
# SAMPLE-параметры
# ============================================================
MAX_PARQUET_FILES = 3
ID_SAMPLE_SIZE = 300_000

CATEGORICAL_OHE_SUM_COLS = [
    "enc_loans_credit_status",
    "enc_loans_credit_type",
]
NUMERIC_BASE_COLS = [
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_total_overdue",
    "pre_loans_max_overdue_sum",
    "pre_loans_credit_cost_rate",
    "pre_loans5",
    "pre_loans530",
    "pre_loans3060",
    "pre_loans6090",
    "pre_loans90",
]

# ============================================================
# Загрузка parquet (column pruning, без дат)
# ============================================================
def get_parquet_files(data_dir: Path = RAW_DATA_DIR) -> List[Path]:
    files = sorted(data_dir.glob("*.pq"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return files[:MAX_PARQUET_FILES]


def load_parquet_iter(files: Iterable[Path]) -> Iterable[pd.DataFrame]:
    for file in files:
        df = pd.read_parquet(file)

        keep_cols = (
            ["id", "rn"]
            + [c for c in df.columns if c.startswith("enc_paym_")]
            + [c for c in NUMERIC_BASE_COLS if c in df.columns]
#            + [c for c in df.select_dtypes(include="number").columns]
 #           + [c for c in CATEGORICAL_MODE_COLS if c in df.columns]
        )

        keep_cols = list(dict.fromkeys(keep_cols))
        yield df[keep_cols]


# ============================================================
# Numeric агрегации (id-level)
# ============================================================
def aggregate_numeric_except_paym(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in NUMERIC_BASE_COLS
        if c in df.columns
    ]

    if not numeric_cols:
        return df[["id"]].drop_duplicates()

    out = (
        df.groupby("id", as_index=False)[numeric_cols]
        .max()
    )
    out.columns = ["id"] + [f"{c}_max" for c in numeric_cols]
    return out


def aggregate_paym_max(df: pd.DataFrame) -> pd.DataFrame:
    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]

    if not paym_cols:
        return df[["id"]].drop_duplicates()

    return (
        df.groupby("id", as_index=False)[paym_cols]
        .max()
    )


# ============================================================
# Категориальные: MODE / LAST
# ============================================================
def aggregate_categorical_mode(
    df: pd.DataFrame,
    cols: list[str],
    id_col: str = "id",
) -> pd.DataFrame:
    out = df[[id_col]].drop_duplicates().copy()

    for col in cols:
        if col not in df.columns:
            continue

        tmp = (
            df[[id_col, col]]
            .dropna()
            .groupby(id_col)[col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1])
            .reset_index()
            .rename(columns={col: f"{col}_mode"})
        )

        out = out.merge(tmp, on=id_col, how="left")

    return out

def aggregate_categorical_ohe_sum(
    df: pd.DataFrame,
    cols: list[str],
    id_col: str = "id",
) -> pd.DataFrame:
    """
    OHE категорий + SUM по id.
    Даёт счётчики событий (frequency encoding).
    """
    out = df[[id_col]].drop_duplicates().copy()

    for col in cols:
        if col not in df.columns:
            continue

        dummies = pd.get_dummies(
            df[[id_col, col]],
            columns=[col],
            prefix=col,
            dtype="int8",
        )

        agg = (
            dummies
            .groupby(id_col)
            .sum()
            .reset_index()
        )

        out = out.merge(agg, on=id_col, how="left")

    return out


# ============================================================
# SEQUENCE-FE по rn (вместо временных фич)
# ============================================================
def build_sequence_features(
    df: pd.DataFrame,
    id_col: str = "id",
    rn_col: str = "rn",
) -> pd.DataFrame:
    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]

    if rn_col not in df.columns or not paym_cols:
        return df[[id_col]].drop_duplicates()

    g = df.sort_values(rn_col).groupby(id_col)

    # базовые rn-фичи
    out = (
        g[rn_col]
        .agg(
            rn_min="min",
            rn_max="max",
            rn_span=lambda x: x.max() - x.min(),
        )
        .reset_index()
        .set_index(id_col)
    )

    # last paym — есть всегда
    last_paym = g[paym_cols].last().max(axis=1)

    # prev paym — может отсутствовать
    prev_paym = (
        g[paym_cols]
        .nth(-2)
        .max(axis=1)
    )

    # выравниваем по id
    out["paym_last"] = last_paym
    out["paym_prev"] = prev_paym
    out["paym_delta"] = out["paym_last"] - out["paym_prev"]

    return out.reset_index()



# ============================================================
# Сборка base dataset
# ============================================================
def build_base_dataset() -> pd.DataFrame:
    parquet_files = get_parquet_files()

    num_chunks = []
    paym_chunks = []
    #cat_chunks = []
    seq_chunks = []
    ohe_sum_chunks = []

    for chunk in load_parquet_iter(parquet_files):
        num_chunks.append(aggregate_numeric_except_paym(chunk))
        paym_chunks.append(aggregate_paym_max(chunk))
        # cat_chunks.append(
        #     aggregate_categorical_mode(
        #         chunk,
        #         cols=CATEGORICAL_MODE_COLS,
        #         id_col="id",
        #     )
        # )
        # OHE + SUM
        ohe_sum_chunks.append(
            aggregate_categorical_ohe_sum(
                chunk,
                cols=CATEGORICAL_OHE_SUM_COLS,
                id_col="id",
            )
        )
        seq_chunks.append(build_sequence_features(chunk))

    num_df = (
        pd.concat(num_chunks, axis=0)
        .groupby("id", as_index=False)
        .max()
    )

    paym_df = (
        pd.concat(paym_chunks, axis=0)
        .groupby("id", as_index=False)
        .max()
    )

    # cat_df = (
    #     pd.concat(cat_chunks, axis=0)
    #     .groupby("id", as_index=False)
    #     .last()
    # )

    seq_df = (
        pd.concat(seq_chunks, axis=0)
        .groupby("id", as_index=False)
        .last()
    )
    ohe_sum_df = (
        pd.concat(ohe_sum_chunks, axis=0)
        .groupby("id", as_index=False)
        .sum()
    )


    dataset = (
        num_df
        .merge(paym_df, on="id", how="left")
#        .merge(cat_df, on="id", how="left")
        .merge(seq_df, on="id", how="left")
        .merge(ohe_sum_df, on="id", how="left")
    )

    # sample по id
    if len(dataset) > ID_SAMPLE_SIZE:
        dataset = (
            dataset
            .sample(n=ID_SAMPLE_SIZE, random_state=42)
            .reset_index(drop=True)
        )

    return dataset









