from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib as jb

data = pd.read_csv('../data/train.csv')
X, y = data.drop(columns=['Class']), data['Class']

def get_hour(df):
    df['Hour'] = pd.to_datetime(df['Time'], unit='s').dt.hour
    return df

rf_smote_pipeline = Pipeline([
    ('Hour', FunctionTransformer(get_hour)),
    ('Oversampling', SMOTE()),
    ('Model', RandomForestClassifier(random_state=42))
])

rf_pipeline = Pipeline([
    ('Hour', FunctionTransformer(get_hour)),
    ('Model', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

xgb_smote_pipeline = Pipeline([
    ('Hour', FunctionTransformer(get_hour)),
    ('Oversampling', SMOTE()),
    ('Model', XGBClassifier(eval_metric='logloss', random_state=42))
])

xgb_pipeline = Pipeline([
    ('Hour', FunctionTransformer(get_hour)),
    ('Model', XGBClassifier(eval_metric='logloss', random_state=42))
])

models = {
    'RandomForest': rf_pipeline,
    'XGBoost': xgb_pipeline,
    'RandomForest_smote': rf_smote_pipeline,
    'XGBoost_smote': xgb_smote_pipeline
}

param_distributions = {
    'RandomForest': {
        'Model__n_estimators': randint(100, 1000),
        'Model__max_depth': randint(3, 20),
        'Model__min_samples_split': randint(2, 20),
        'Model__min_samples_leaf': randint(1, 20),
        'Model__max_features': uniform(0.3, 0.7),
        'Model__bootstrap': [True, False],
    },
    'RandomForest_smote': {
        'Model__n_estimators': randint(100, 1000),
        'Model__max_depth': randint(3, 20),
        'Model__min_samples_split': randint(2, 20),
        'Model__min_samples_leaf': randint(1, 20),
        'Model__max_features': uniform(0.3, 0.7),
        'Model__bootstrap': [True, False],
    },
    'XGBoost_smote': {
        'Model__n_estimators': randint(100, 2000),
        'Model__learning_rate': loguniform(0.001, 0.3),
        'Model__max_depth': randint(3, 12),
        'Model__min_child_weight': loguniform(1, 100),
        'Model__gamma': loguniform(1e-3, 10),
        'Model__subsample': uniform(0.5, 0.5),
        'Model__colsample_bytree': uniform(0.5, 0.5),
        'Model__reg_lambda': loguniform(1e-3, 1000),
        'Model__reg_alpha': loguniform(1e-3, 100)
    },
    'XGBoost': {
        'Model__n_estimators': randint(100, 2000),
        'Model__learning_rate': loguniform(0.001, 0.3),
        'Model__max_depth': randint(3, 12),
        'Model__min_child_weight': loguniform(1, 100),
        'Model__gamma': loguniform(1e-3, 10),
        'Model__subsample': uniform(0.5, 0.5),
        'Model__colsample_bytree': uniform(0.5, 0.5),
        'Model__reg_lambda': loguniform(1e-3, 1000),
        'Model__reg_alpha': loguniform(1e-3, 100),
        'Model__scale_pos_weight': loguniform(1, (y == 0).sum() * 100 / len(y))
    }
}

for name, model in models.items():
    random_search = RandomizedSearchCV(
        model,
        param_distributions[name],
        n_iter=20,
        scoring='average_precision',
        cv=TimeSeriesSplit(n_splits=3, max_train_size=50000, test_size=10000, gap=1000),
        n_jobs=-1,
        random_state=42,
        refit=True
    )
    random_search.fit(X, y)
    best_model = [random_search.best_estimator_, random_search.best_score_]
    models[name] = best_model
    print(f"Best parameters for {name}: {random_search.best_params_}")
    jb.dump(best_model, f'../models/{name}_best_model.joblib')