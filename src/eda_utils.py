import pandas as pd
import numpy as np


def get_high_freq_columns(df: pd.DataFrame, threshold: int) -> list:
    cols_frequency = {
        col: df[col].value_counts(dropna=False, normalize=True).to_dict()
        for col in df.columns
    }
    high_freq_columns = [
        col
        for col, values in cols_frequency.items()
        if sorted(values.values(), reverse=True)[0] > threshold
    ]
    return high_freq_columns


def get_corr(df: pd.DataFrame, method: str = "pearson") -> dict:
    col_correlations = abs(df.corr(method=method))
    col_correlations.loc[:, :] = np.tril(col_correlations)
    cor_pairs = col_correlations.stack()
    return cor_pairs.to_dict()


def get_high_corr_columns(df: pd.DataFrame, threshold: int) -> list:
    high_corr_columns = [
        cols
        for cols, corr in get_corr(df).items()
        if (corr > threshold) and (cols[0] != cols[1])
    ]
    return high_corr_columns


def missing_values_percentage(df: pd.DataFrame) -> pd.Series:
    return ((df[df.columns[df.isna().any()]].isna().sum() / len(df)) * 100).sort_values(
        ascending=False
    )
