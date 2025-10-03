# Credit Card Fraud Detection

This project analyzes **credit card fraud data** and builds machine learning models to detect fraudulent transactions. It uses classical ML methods and advanced models such as XGBoost, with a focus on handling **imbalanced datasets** effectively.

**Key dependencies:**

* scikit-learn
* numpy
* scipy
* matplotlib
* seaborn
* xgboost
* joblib
* pandas
* kaggle
* pathlib
* zipfile
* imbalanced-learn (imblearn)

In order to reproduce the results, run the runner.py file in the src folder.

## 📊 Features

* Exploratory data analysis (EDA)
* Preprocessing and feature scaling
* Handling class imbalance (SMOTE, undersampling)
* Model training and evaluation (Logistic Regression, Random Forest, XGBoost)
* Precision-Recall analysis and ROC curves
* Model saving/loading with `joblib`

---

## 📈 Results

* Evaluation metrics include **Precision, Recall, F1-score, ROC AUC, and PR AUC**.
* Trade-offs between precision and recall are explored in the context of fraud detection.

---

## 📜 License

This project is released under the **MIT License**.

The dataset is licensed separately by the original authors:

* **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Dataset License**: [Database: Open Database, Contents: Database Contents](http://opendatacommons.org/licenses/dbcl/1.0/)
