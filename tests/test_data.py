import pandas as pd

from src.data import clean_data


def test_clean_data_removes_missing_total_charges():

    df = pd.DataFrame(
        {
            "Churn": ["Yes", "No"],
            "TotalCharges": ["100.5", ""],
        }
    )

    cleaned = clean_data(df)

    assert len(cleaned) == 1
    assert cleaned["TotalCharges"].dtype != object

