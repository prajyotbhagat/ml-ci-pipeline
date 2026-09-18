import pandas as pd

from src.train import build_pipeline


def test_model_can_train():

    X = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 36],
            "MonthlyCharges": [50, 60, 70, 80],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Two year",
            ],
        }
    )

    y = [1, 0, 0, 0]

    pipeline = build_pipeline(X)

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
