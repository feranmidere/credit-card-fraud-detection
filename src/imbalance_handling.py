import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/train.csv').drop(columns=['Time'])

X = train.drop(columns=['Class'])
y = train['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

train_resampled = pd.concat([X_resampled, y_resampled], axis=1)
train_resampled.to_csv('../data/train_resampled.csv', index=False)