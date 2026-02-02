# генерация признаков

import numpy as np
import pandas as pd


def add_overdue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group A: агрегаты по просрочкам.
    Работает с агрегированными (_max) колонками.
    """
    df = df.copy()

    # ожидаемые агрегированные колонки по просрочкам
    overdue_cols = [
        "pre_loans5_max",
        "pre_loans530_max",
        "pre_loans3060_max",
        "pre_loans6090_max",
        "pre_loans90_max",
    ]

    cols = [c for c in overdue_cols if c in df.columns]

    # 1. Общее количество типов просрочек
    if cols:
        df["overdue_cnt_total"] = df[cols].sum(axis=1)
    else:
        df["overdue_cnt_total"] = 0

    # 2. Длинные просрочки (60+)
    df["overdue_cnt_long"] = (
        df.get("pre_loans6090_max", 0) + df.get("pre_loans90_max", 0)
    )

    # 3. Короткие просрочки (≤30)
    df["overdue_cnt_short"] = (
        df.get("pre_loans5_max", 0) + df.get("pre_loans530_max", 0)
    )

    # 4. Доля длинных просрочек
    df["share_long_overdue"] = (
        df["overdue_cnt_long"] / (df["overdue_cnt_total"] + 1)
    )

    # 5. Были ли вообще просрочки
    df["has_any_overdue"] = (df["overdue_cnt_total"] > 0).astype(int)

    # 6. Были ли тяжёлые просрочки (90+)
    df["has_90plus_overdue"] = (df.get("pre_loans90_max", 0) > 0).astype(int)

    # 7. Тяжесть просрочек (детерминированно, без idxmax)
    df["max_overdue_bucket"] = (
        df.get("pre_loans5_max", 0) * 0
        + df.get("pre_loans530_max", 0) * 1
        + df.get("pre_loans3060_max", 0) * 2
        + df.get("pre_loans6090_max", 0) * 3
        + df.get("pre_loans90_max", 0) * 4
    )

    # 8. Поведенческий флаг по платежам
    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]
    if paym_cols:
        df["has_any_bad_payment"] = (df[paym_cols].max(axis=1) > 0).astype(int)
    else:
        df["has_any_bad_payment"] = 0

    return df


def add_limit_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group B: отношения к кредитному лимиту.
    """

    df = df.copy()

    limit = df["pre_loans_credit_limit_max"] + 1.0
    outstanding = df["pre_loans_outstanding_max"]
    overdue = df["pre_loans_total_overdue_max"]

    df["util_ratio"] = outstanding / limit
    df["overdue_ratio"] = overdue / limit

    return df

def add_payment_discipline_features(
    df: pd.DataFrame,
    bad_threshold: int = 2,
) -> pd.DataFrame:
    """
    Feature Group C: платёжная дисциплина по enc_paym_*.

    bad_threshold:
        значения >= bad_threshold считаем плохими платежами
    """

    df = df.copy()

    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]
    if not paym_cols:
        return df

    paym_values = df[paym_cols].values

    # 1. Количество плохих платежей
    bad_mask = paym_values >= bad_threshold
    df["cnt_bad_payments"] = bad_mask.sum(axis=1)

    # 2. Доля плохих платежей
    df["share_bad_payments"] = df["cnt_bad_payments"] / len(paym_cols)

    # 3. Худший статус платежа
    df["max_paym_status"] = paym_values.max(axis=1)

    # 4. Последний платёж плохой
    # считаем, что enc_paym_0 — самый свежий
    df["last_payment_bad"] = (
        df["enc_paym_0"] >= bad_threshold
    ).astype(int)

    # 5. Максимальная серия подряд плохих платежей
    # считаем по строкам
    max_streak = []
    for row in bad_mask:
        streak = 0
        best = 0
        for v in row:
            if v:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        max_streak.append(best)

    df["max_bad_streak"] = max_streak

    return df

def add_credit_timeline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group D: временные признаки жизненного цикла кредита.
    """

    df = df.copy()

    # Защита от деления на 0
    pterm = df["pre_pterm"] + 1.0

    # 1. Доля уже прошедшего срока кредита
    df["credit_age_ratio"] = df["pre_since_opened"] / pterm

    # 2. Доля оставшегося срока до планового закрытия
    df["time_to_close_ratio"] = df["pre_till_pclose"] / pterm

    # 3. Новый кредит (ранняя стадия)
    df["is_new_credit"] = (df["credit_age_ratio"] < 0.2).astype(int)

    # 4. Кредит близок к закрытию
    df["is_near_close"] = (df["time_to_close_ratio"] < 0.1).astype(int)

    # 5. Долгий кредит
    df["is_long_credit"] = (df["pre_pterm"] > df["pre_pterm"].median()).astype(int)

    # 6. Фактически закрыт
    df["is_closed_fact"] = (df["fclose_flag"] == 1).astype(int)

    return df
