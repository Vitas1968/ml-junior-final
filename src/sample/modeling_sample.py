# обучение моделей

from catboost import CatBoostClassifier


from catboost import CatBoostClassifier


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
) -> CatBoostClassifier:
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        thread_count=-1,
        verbose=100,
        early_stopping_rounds=200,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    return model

