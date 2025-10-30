# src/feature_engineering.py
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_TARGET = "default"

def get_feature_target(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y

def train_test_split_data(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = get_feature_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    return X_train, X_test, y_train, y_test

def scale_numeric(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler

def get_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in exclude]