import kaggle
import pathlib
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

kaggle.api.dataset_download_file('mlg-ulb/creditcardfraud', 'creditcard.csv', path='../data')
with zipfile.ZipFile('../data/creditcard.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')
pathlib.Path('../data/creditcard.csv.zip').unlink()

data = pd.read_csv('../data/creditcard.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Class'])
train, val = train_test_split(train, test_size=0.25, random_state=42, stratify=train['Class'])
train.to_csv('../data/train.csv', index=False)
val.to_csv('../data/val.csv', index=False)
test.to_csv('../data/test.csv', index=False)