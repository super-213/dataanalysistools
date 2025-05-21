from typing import Literal
import pandas as pd

def impute(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "mode", "constant"] = "mean",
    constant_value: float | int | str | None = None,
) -> pd.DataFrame:
    """
    通用缺失值填补函数。

    Parameters
    ----------
    df : DataFrame
        原始数据
    strategy : {"mean", "median", "mode", "constant"}
        填补策略
    constant_value : 任意
        strategy="constant" 时使用

    Returns
    -------
    DataFrame
        填补后的副本
    """
    res = df.copy()
    for col in res.select_dtypes(include="number"):
        if strategy == "mean":
            res[col] = res[col].fillna(res[col].mean())
        elif strategy == "median":
            res[col] = res[col].fillna(res[col].median())
        elif strategy == "mode":
            res[col] = res[col].fillna(res[col].mode().iloc[0])
        elif strategy == "constant":
            res[col] = res[col].fillna(constant_value)
    return res