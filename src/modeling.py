# обучение моделей

import lightgbm as lgb

def train_model(X_train, y_train) -> lgb.LGBMClassifier:
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=128,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

