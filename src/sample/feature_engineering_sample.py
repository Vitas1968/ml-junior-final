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

def add_overdue_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Усиление overdue-сигнала:
    работаем ТОЛЬКО с *_max, без новых агрегаций и без памяти.
    """
    df = df.copy()

    # ожидаемые колонки (если каких-то нет — считаем их нулями)
    c5   = df.get("pre_loans5_max", 0)
    c30  = df.get("pre_loans530_max", 0)
    c60  = df.get("pre_loans3060_max", 0)
    c90  = df.get("pre_loans6090_max", 0)
    c90p = df.get("pre_loans90_max", 0)

    # 1) максимальная глубина просрочки (0..4)
    df["overdue_max_bucket"] = (
        (c5   > 0).astype(int) * 1 +
        (c30  > 0).astype(int) * 2 +
        (c60  > 0).astype(int) * 3 +
        (c90  > 0).astype(int) * 4 +
        (c90p > 0).astype(int) * 5
    )

    # 2) эскалация: были ли разные уровни (структура, а не сумма)
    df["overdue_levels_cnt"] = (
        (c5   > 0).astype(int) +
        (c30  > 0).astype(int) +
        (c60  > 0).astype(int) +
        (c90  > 0).astype(int) +
        (c90p > 0).astype(int)
    )

    # 3) флаг тяжёлой просрочки (60+)
    df["has_overdue_60_plus"] = ((c60 > 0) | (c90 > 0) | (c90p > 0)).astype(int)

    # 4) флаг экстремальной просрочки (90+)
    df["has_overdue_90_plus"] = (c90p > 0).astype(int)

    # 5) нелинейная тяжесть (взвешенно, без mean)
    df["overdue_severity_weighted"] = (
        1 * (c5   > 0).astype(int) +
        2 * (c30  > 0).astype(int) +
        3 * (c60  > 0).astype(int) +
        4 * (c90  > 0).astype(int) +
        5 * (c90p > 0).astype(int)
    )

    return df


def add_payment_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Усиленные поведенческие признаки на базе enc_paym_*_max.
    Лёгкие по памяти, сильные по сигналу.
    """
    df = df.copy()

    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]

    if not paym_cols:
        # если вдруг нет paym — просто выходим
        return df

    paym_max = df[paym_cols].max(axis=1)

    # 1️⃣ Был ли вообще плохой платёж
    df["paym_any_bad"] = (paym_max > 0).astype(int)

    # 2️⃣ Максимальная тяжесть
    df["paym_max_severity"] = paym_max

    # 3️⃣ Кол-во типов с тяжестью >= 2
    df["paym_bad_cnt_ge_2"] = (df[paym_cols] >= 2).sum(axis=1)

    # 4️⃣ Кол-во типов с тяжестью >= 3
    df["paym_bad_cnt_ge_3"] = (df[paym_cols] >= 3).sum(axis=1)

    # 5️⃣ Взвешенная тяжесть (усиливает редкие пики)
    df["paym_severity_weighted"] = (
        (df[paym_cols] >= 1).sum(axis=1)
        + 2 * (df[paym_cols] >= 2).sum(axis=1)
        + 3 * (df[paym_cols] >= 3).sum(axis=1)
    )

    # 6️⃣ Дискретный bucket риска
    df["paym_bad_bucket"] = (
        (paym_max >= 1).astype(int)
        + (paym_max >= 2).astype(int)
        + (paym_max >= 3).astype(int)
    )

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

def add_payment_discipline_features(df: pd.DataFrame) -> pd.DataFrame:
    paym_cols = [c for c in df.columns if c.startswith("enc_paym_")]
    if not paym_cols:
        return df

    bad_threshold = 3

    df["paym_bad_cnt"] = (df[paym_cols] >= bad_threshold).sum(axis=1)
    df["paym_bad_ratio"] = df["paym_bad_cnt"] / len(paym_cols)

    df["paym_max"] = df[paym_cols].max(axis=1)
    df["paym_mean"] = df[paym_cols].mean(axis=1)

    return df


def add_credit_timeline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки временной структуры кредита.
    Работает ТОЛЬКО с агрегированными колонками *_mean / *_max.
    """
    # используем агрегированные версии
    pterm = df.get("pre_pterm_mean")
    fterm = df.get("pre_fterm_mean")
    since_opened = df.get("pre_since_opened_mean")

    # если ключевых колонок нет — просто выходим
    if pterm is None or fterm is None or since_opened is None:
        return df

    eps = 1.0

    df["credit_term_ratio"] = fterm / (pterm + eps)
    df["credit_age_ratio"] = since_opened / (pterm + eps)
    df["credit_overdue_term_ratio"] = (fterm - pterm) / (pterm + eps)

    return df

