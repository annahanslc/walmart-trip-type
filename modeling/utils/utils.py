import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features.base_features import groupby_visitnumber


def import_data(filepath):
    return pd.read_csv(filepath, dtype={"Upc": str})


def drop_visitnumber(df):
    # Drop the VisitNumber column
    df.drop(columns="VisitNumber", inplace=True)
    return df


def pre_preprocess(df):
    # Drop VisitNumber
    df = drop_visitnumber(df)

    # Use LabelEncoder to encode the labels
    le = LabelEncoder()
    df["triptype"] = le.fit_transform(df["triptype"])

    # Train test split
    X = df.drop(columns="triptype").copy()
    y = df["triptype"].copy()

    # Create a test (holdout) set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(
        f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}"
    )

    return X_train, X_test, y_train, y_test
