"""
This program should predict the laptop price of my current laptop and help to decide whether I paid too much.

Author: Youheng Lü
"""

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from xgboost import XGBRegressor


def get_score(n_estimators: int, learning_rate: float) -> float:
    """
    Returns the average MAE over 3 CV folds of random forest model

    Parameters
        n_estimators  -- number of trees in the forest
        learning_rate -- TODO

    Returns
        float -- TODO
    """

    # Define model
    model = XGBRegressor(
        random_state=0,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores = -1 * cross_val_score(clf, X, y, cv=3, scoring="neg_mean_absolute_error")
    return scores.mean()


if __name__ == "__main__":
    # Read data
    real_path = os.path.realpath(__file__)
    os.chdir(real_path + "/..")

    # Found the encoding online
    X_full = pd.read_csv("laptop_price.csv", encoding="ISO-8859-1")
    X_test_full = pd.read_csv("my_laptop.csv")
    print(X_full.head())
    print(X_full.columns)

    # Split into two different kinds of columns
    target_column = "Price_euros"

    # drop rows with missing target
    X_full.dropna(axis=0, subset=[target_column], inplace=True)
    y = X_full[target_column]
    X_full.drop([target_column], axis=1, inplace=True)

    # Categorical columns with cardinality < 10
    categorical_cols = [
        cname
        for cname in X_full.columns
        if X_full[cname].nunique() < 10 and X_full[cname].dtype == "object"
    ]

    # Numerical columns
    numerical_cols = [
        cname for cname in X_full.columns if X_full[cname].dtype in ["int64", "float64"]
    ]

    # Keep only selected columns
    my_cols = categorical_cols + numerical_cols
    X = X_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()
    print("Removed columns = ", set(X_full.columns) - set(X.columns))

    # Preprocessor
    # Preprocessor for numerical data
    numerical_transformer = SimpleImputer(strategy="mean")

    # Preprocessor for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    possible_n_estimators = list(range(50, 401, 50))
    learning_rate = 0.05

    results = [get_score(p, learning_rate) for p in tqdm(possible_n_estimators)]
    fig, ax = plt.subplots(1)
    ax.plot(possible_n_estimators, results)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("MAE")

    # Find best learning rate
    possible_n_estimators = 50
    learning_rate = np.arange(0.01, 0.1, 0.01)

    results = [get_score(possible_n_estimators, l) for l in tqdm(learning_rate)]
    fig, ax = plt.subplots(1)
    ax.plot(learning_rate, results)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("MAE")

    print(min(results))

    # Final best model
    model = XGBRegressor(
        random_state=0,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.05,
    )
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    final_pipeline.fit(X_full, y)
    predictions = final_pipeline.predict(X_test_full)

    # save output
    print("Predicted Price:", predictions)
    print("Used columns = ")
    print(X_test.columns)
