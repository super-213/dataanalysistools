import pandas as pd
from typing import Iterable

def drop_columns(
    df: pd.DataFrame,
    cols_to_drop: Iterable[str] = None,
    dtypes_to_drop: Iterable[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    删除 DataFrame 中的指定列或指定类型的列。

    参数
    ----
    df : pd.DataFrame
        要处理的数据框

    cols_to_drop : 可迭代[str], optional
        要删除的列名列表（如 ['id', 'name']）

    dtypes_to_drop : 可迭代[str], optional
        要删除的数据类型（如 ['object', 'datetime64[ns]']）

    verbose : bool, default False
        是否打印删除的列名

    返回
    ----
    pd.DataFrame
        删除列后的副本
    """
    to_drop = set()

    # 通过列名删除
    if cols_to_drop:
        existing_cols = set(df.columns)
        to_drop.update(set(cols_to_drop) & existing_cols)

    # 通过数据类型删除
    if dtypes_to_drop:
        for dtype in dtypes_to_drop:
            matched_cols = df.select_dtypes(include=[dtype]).columns
            to_drop.update(matched_cols)

    if verbose:
        print(f"Dropping columns: {sorted(to_drop)}")

    return df.drop(columns=to_drop)