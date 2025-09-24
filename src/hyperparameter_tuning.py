from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import xgboost as xgb
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib as jb

data = pd.read_csv('../data/train_resampled.csv')
X, y = data.drop(columns=['Class']), data['Class']
X_sample, _, y_sample, _ = train_test_split(
    X, y, train_size=0.2, stratify=y, random_state=42
)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42)
}

param_distributions = {
    'RandomForest': {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': uniform(0.3, 0.7),
        'bootstrap': [True, False],
    },
    'XGBoost': {
        'n_estimators': randint(100, 2000),
        'learning_rate': loguniform(0.001, 0.3),
        'max_depth': randint(3, 12),
        'min_child_weight': loguniform(1, 100),
        'gamma': loguniform(1e-3, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'reg_lambda': loguniform(1e-3, 1000),
        'reg_alpha': loguniform(1e-3, 100),
    }
}

for name, model in models.items():
    random_search = RandomizedSearchCV(
        model,
        param_distributions[name],
        n_iter=50,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(X_sample, y_sample)
    best_model = random_search.best_estimator_
    models[name] = best_model
    print(f"Best parameters for {name}: {random_search.best_params_}")
    jb.dump(best_model, f'../models/{name}_best_model.joblib')