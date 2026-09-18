import pandas as pd


TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    """Load the customer churn dataset."""
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer data."""

    df = df.copy()

    # TotalCharges contains some empty strings
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna()

    return df
