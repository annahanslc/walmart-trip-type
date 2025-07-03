import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer

from sklearn import set_config

set_config(transform_output="pandas")


class PreprocessingV2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_features = None
        self.cat_features = None
        self.pipeline = None

    def fit(self, X, y=None):
        self.num_features = X.select_dtypes(include="number").columns.to_list()
        self.cat_features = X.select_dtypes(include="object").columns.to_list()

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
        X_trans = self.pipeline.transform(X)
        return X_trans
