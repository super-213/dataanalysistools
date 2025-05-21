import pandas as pd
import numpy as np
from typing import List, Union, Optional

def bin_continuous_variable(
    df: pd.DataFrame,
    column: str,
    method: str = 'equal_width',  # 'equal_width', 'equal_freq', 'custom'
    bins: Union[int, List[float]] = 5,
    labels: Optional[List[str]] = None,
    return_interval: bool = True,
    new_col_name: Optional[str] = None,
    drop_original: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    对连续变量进行分箱处理。

    参数:
        df (pd.DataFrame): 输入 DataFrame。
        column (str): 要分箱的列名。
        method (str): 分箱方法，'equal_width'（等宽）、'equal_freq'（等频）、'custom'（自定义）。
        bins (int or List[float]): 分箱数量或自定义边界。
        labels (List[str], optional): 分箱标签。如果不传将使用默认区间或数字编号。
        return_interval (bool): 是否显示区间（仅对默认 label 生效）。
        new_col_name (str): 分箱后新列的名称，默认为原列名+"_binned"。
        drop_original (bool): 是否删除原始列。
        inplace (bool): 是否在原 df 上修改。

    返回:
        pd.DataFrame: 处理后的 DataFrame。
    """
    df_new = df if inplace else df.copy()

    if new_col_name is None:
        new_col_name = column + "_binned"

    if method == 'equal_width':
        df_new[new_col_name] = pd.cut(df_new[column], bins=bins, labels=labels if labels else None)

    elif method == 'equal_freq':
        df_new[new_col_name] = pd.qcut(df_new[column], q=bins, labels=labels if labels else None, duplicates='drop')

    elif method == 'custom':
        if not isinstance(bins, (list, np.ndarray)):
            raise ValueError("自定义分箱时，bins 应为边界列表。")
        df_new[new_col_name] = pd.cut(df_new[column], bins=bins, labels=labels if labels else None, include_lowest=True)

    else:
        raise ValueError("method 取值应为 ['equal_width', 'equal_freq', 'custom']")

    if drop_original:
        df_new.drop(columns=[column], inplace=True)

    return df_new