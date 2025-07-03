import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config
from sklearn.compose import ColumnTransformer

from sklearn import set_config

set_config(transform_output="pandas")


class PreprocessingV1(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_features = None
        self.cat_features = None
        self.pipeline = None
        self.expected_columns = None

    def fit(self, X, y=None):
        X = X.copy()
        if self.num_features is None:
            self.num_features = X.select_dtypes(include="number").columns.to_list()
        if self.cat_features is None:
            self.cat_features = X.select_dtypes(include="object").columns.to_list()

        # Save columns expected for future transforms
        self.expected_columns = X.columns.tolist()

        self.pipeline = ColumnTransformer(
            [
                ("num", StandardScaler(), self.num_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.cat_features,
                ),
            ]
        )
        self.pipeline.fit(X)
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for col in self.expected_columns:
            if col not in X.columns:
                X[col] = np.nan if col in self.num_features else "missing"

        # Ensure column order is consistent with training
        X = X[self.expected_columns]

        return self.pipeline.transform(X)
