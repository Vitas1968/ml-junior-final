# обучение моделей

from catboost import CatBoostClassifier


def train_model(X_train, y_train) -> CatBoostClassifier:
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False,
    )

    model.fit(X_train, y_train)
    return model
