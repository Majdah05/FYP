from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IDSDataInterface:
    """Interface for later IDS integration.

    The exact dataset schema is not known in advance. The caller must provide the
    feature columns, target column, and any categorical columns. The interface
    then fits preprocessing on the training split only, avoiding leakage.
    """

    def __init__(self, dataset_path: str | Path, target_column: str, categorical_columns=None, numeric_columns=None):
        self.dataset_path = Path(dataset_path)
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.numeric_columns = numeric_columns or []
        if not self.dataset_path.exists():
            raise FileNotFoundError(f'IDS dataset not found at {self.dataset_path}')

    def load(self):
        df = pd.read_csv(self.dataset_path)
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing from the IDS dataset.")
        if not self.numeric_columns and not self.categorical_columns:
            self.numeric_columns = [c for c in df.columns if c != self.target_column and pd.api.types.is_numeric_dtype(df[c])]
            self.categorical_columns = [c for c in df.columns if c != self.target_column and c not in self.numeric_columns]
        return df

    def build_preprocessor(self):
        numeric_features = self.numeric_columns or []
        categorical_features = self.categorical_columns or []
        transformers = []
        if numeric_features:
            transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features))
        if categorical_features:
            transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features))
        return ColumnTransformer(transformers=transformers, remainder='drop')
