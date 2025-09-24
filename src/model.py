import joblib as jb
from sklearn.metrics import *
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train = pd.read_csv('../data/train_resampled.csv')
X_train, y_train = train.drop(columns=['Class']), train['Class']
val = pd.read_csv('../data/val.csv').drop(columns=['Time'])
X_val, y_val = val.drop(columns=['Class']), val['Class']

xgb_model = jb.load('../models/XGBoost_best_model.joblib')
rf_model = jb.load('../models/RandomForest_best_model.joblib')

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
rf_model.fit(X_train, y_train)

jb.dump(xgb_model, '../models/XGBoost_final_model.joblib')
jb.dump(rf_model, '../models/RandomForest_final_model.joblib')