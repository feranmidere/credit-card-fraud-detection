import kaggle
import pathlib
import zipfile
import pandas as pd

kaggle.api.dataset_download_file('mlg-ulb/creditcardfraud', 'creditcard.csv', path='../data')
with zipfile.ZipFile('../data/creditcard.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')
pathlib.Path('../data/creditcard.csv.zip').unlink()

data = pd.read_csv('../data/creditcard.csv')
train = data[:round(0.75 * len(data))]
test = data[round(0.75 * len(data)):]

train.to_csv('../data/train.csv', index=False)
test.to_csv('../data/test.csv', index=False)