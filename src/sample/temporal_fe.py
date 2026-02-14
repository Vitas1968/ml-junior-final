import pandas as pd
import numpy as np

def build_temporal_features(
    df: pd.DataFrame,
    date_col: str,
    id_col: str = "id",
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Временные фичи по окнам 30/90/180 дней.
    df — транзакционный уровень.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if ref_date is None:
        ref_date = df[date_col].max()

    windows = {
        "30d": 30,
        "90d": 90,
        "180d": 180,
    }

    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]

    out = []

    for w_name, w_days in windows.items():
        start = ref_date - pd.Timedelta(days=w_days)
        wdf = df[df[date_col] >= start]

        agg = {
            "events_cnt": (date_col, "count"),
            "paym_max": (paym_cols, "max"),
        }

        g = wdf.groupby(id_col)

        feat = g.agg(**{
            f"events_cnt_{w_name}": (date_col, "count"),
        }).reset_index()

        if paym_cols:
            paym_max = g[paym_cols].max().max(axis=1)
            feat[f"paym_max_{w_name}"] = paym_max.values

            feat[f"paym_cnt_ge_2_{w_name}"] = (
                g[paym_cols].apply(lambda x: (x >= 2).sum().sum())
                .values
            )

        last_event = g[date_col].max()
        feat[f"recency_{w_name}"] = (ref_date - last_event).dt.days.values

        out.append(feat)

    # merge окон
    res = out[0]
    for f in out[1:]:
        res = res.merge(f, on=id_col, how="left")

    # динамика
    if "paym_max_30d" in res and "paym_max_90d" in res:
        res["delta_paym_30_90"] = res["paym_max_30d"] - res["paym_max_90d"]

    return res
