import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_data, clean_data


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"


def build_pipeline(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=1000
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main():

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    pipeline = build_pipeline(X)

    pipeline.fit(X, y)

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
