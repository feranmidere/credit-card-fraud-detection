import kaggle
import pathlib
import zipfile
import pandas as pd
from .config import DATA_DIR, MODELS_DIR, PROJECT_ROOT

def main():
    kaggle.api.dataset_download_file('mlg-ulb/creditcardfraud', 'creditcard.csv', path=DATA_DIR)
    data_filepath = DATA_DIR / 'creditcard.csv'
    with zipfile.ZipFile(data_filepath, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    pathlib.Path(data_filepath).unlink()

    data = pd.read_csv(data_filepath)
    train = data[:round(0.75 * len(data))]
    test = data[round(0.75 * len(data)):]

    train.to_csv(DATA_DIR / 'train.csv', index=False)
    test.to_csv(DATA_DIR / 'test.csv', index=False)

if __name__ == '__main__':
    main()