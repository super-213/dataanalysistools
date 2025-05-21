import pandas as pd
import numpy as np
from scipy.stats import zscore

def detect_outliers_zscore(
    df: pd.DataFrame,
    cols: list = None,
    threshold: float = 3.0,
    method: str = "both",         # 'both', 'high', 'low'
    return_zscore: bool = False,
    drop: bool = False,
    verbose: bool = False
) -> pd.DataFrame | tuple:
    """
    使用 Z-Score 方法检测并可剔除异常值。

    参数
    ----
    df : pd.DataFrame
        原始数据

    cols : list[str], optional
        要检测的列（默认全部数值列）

    threshold : float
        Z 分数阈值，超过即视为异常

    method : {'both', 'high', 'low'}
        检测异常值方向：
            - 'both': 正负两侧都检测（默认）
            - 'high': 仅检测高值异常（z > threshold）
            - 'low':  仅检测低值异常（z < -threshold）

    return_zscore : bool
        是否返回异常值的 Z 分数（附加在结果中）

    drop : bool
        是否直接剔除异常值行（返回清洗后的 DataFrame）

    verbose : bool
        是否输出每列异常值数量

    返回
    ----
    如果 drop=False 且 return_zscore=False:
        返回异常值的行（pd.DataFrame）

    如果 drop=False 且 return_zscore=True:
        返回包含 Z 分数的异常值 DataFrame

    如果 drop=True:
        返回剔除后的 DataFrame（无异常值）
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = pd.Series(False, index=df.index)
    zscore_dict = {}

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std

        if method == 'both':
            mask = np.abs(z_scores) > threshold
        elif method == 'high':
            mask = z_scores > threshold
        elif method == 'low':
            mask = z_scores < -threshold
        else:
            raise ValueError("method must be one of ['both', 'high', 'low']")

        outlier_mask |= mask
        zscore_dict[col] = z_scores

        if verbose:
            print(f"[{col}] 异常值数量: {mask.sum()}")

    if drop:
        return df[~outlier_mask]

    outliers = df[outlier_mask]
    if return_zscore:
        zscore_df = pd.DataFrame({col: zscore_dict[col] for col in cols})
        outliers = outliers.copy()
        for col in cols:
            outliers[f"{col}_z"] = zscore_df.loc[outliers.index, col]

    return outliers




def detect_outliers_iqr(
    df: pd.DataFrame,
    cols: list = None,
    multiplier: float = 1.5,
    method: str = "both",  # "both", "high", "low"
    return_bounds: bool = False,
    drop: bool = False,
    verbose: bool = False
) -> pd.DataFrame | tuple:
    """
    使用 IQR（四分位数）方法检测异常值，并可选择剔除。

    参数
    ----
    df : pd.DataFrame
        原始数据

    cols : list[str], optional
        要检测的列（默认全部数值列）

    multiplier : float
        IQR 倍数阈值（默认 1.5）

    method : {"both", "high", "low"}
        异常方向控制：
            - "both": 上下两端都检测
            - "high": 仅检测上异常（值 > Q3 + 1.5×IQR）
            - "low" : 仅检测下异常（值 < Q1 - 1.5×IQR）

    return_bounds : bool
        是否返回每列的上下界（Q1 和 Q3 基础上的判断线）

    drop : bool
        是否直接剔除异常值行（返回清洗后的 df）

    verbose : bool
        是否打印每列异常数量

    返回
    ----
    如果 drop=False:
        返回异常值 DataFrame

    如果 drop=True:
        返回剔除异常值后的 DataFrame

    如果 return_bounds=True:
        返回 (异常值DataFrame或清洗后df, bounds字典)
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = pd.Series(False, index=df.index)
    bounds = {}

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        bounds[col] = (lower, upper)

        if method == "both":
            mask = (df[col] < lower) | (df[col] > upper)
        elif method == "high":
            mask = df[col] > upper
        elif method == "low":
            mask = df[col] < lower
        else:
            raise ValueError("method must be one of ['both', 'high', 'low']")

        outlier_mask |= mask

        if verbose:
            print(f"[{col}] 异常值数量: {mask.sum()}")

    if drop:
        cleaned_df = df[~outlier_mask]
        return (cleaned_df, bounds) if return_bounds else cleaned_df
    else:
        outliers = df[outlier_mask]
        return (outliers, bounds) if return_bounds else outliers
    

import matplotlib.pyplot as plt
import seaborn as sns
import math

def detect_outliers_boxplot(df, features, max_per_page=9, figsize=(15, 10), dpi=100):
    """
    用箱线图批量可视化异常值

    df: pandas DataFrame

    features: 要检查的特征列表（数值型）

    max_per_page: 每页最多显示的子图数

    figsize: 每页图像尺寸
    
    dpi: 图像清晰度
    """
    total_features = len(features)
    total_pages = math.ceil(total_features / max_per_page)
    
    for page in range(total_pages):
        start_idx = page * max_per_page
        end_idx = min(start_idx + max_per_page, total_features)
        features_subset = features[start_idx:end_idx]
        
        num_subplots = len(features_subset)
        cols = 3
        rows = math.ceil(num_subplots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
        fig.suptitle(f'Boxplot Outlier Detection (Page {page + 1})', fontsize=16)

        # Flatten axes for easy iteration, handle if only 1 row
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        for i, feature in enumerate(features_subset):
            sns.boxplot(y=df[feature], ax=axes[i], color="skyblue")
            axes[i].set_title(feature)
        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def handle_outliers(
    df: pd.DataFrame,
    cols: list = None,
    method: str = 'zscore',         # 'zscore' or 'iqr'
    strategy: str = 'remove',       # 'remove', 'replace_mean', 'replace_median', 'clip'
    z_thresh: float = 3.0,
    iqr_mult: float = 1.5,
    direction: str = 'both',        # 'both', 'high', 'low'
    verbose: bool = False
) -> pd.DataFrame:
    """
    处理异常值：删除、替换或裁剪。

    参数：
    - df : 原始 DataFrame
    - cols : 指定处理列（默认数值列）
    - method : 异常检测方法（zscore 或 iqr）
    - strategy : 处理策略（删除、替换、裁剪）
    - z_thresh : Z-Score 阈值
    - iqr_mult : IQR 倍数
    - direction : 异常检测方向（上下）
    - verbose : 是否打印信息
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    rows_to_remove = pd.Series(False, index=df.index)

    for col in cols:
        if method == 'zscore':
            z_scores = zscore(df[col].dropna())
            z_series = pd.Series(np.nan, index=df[col].index)
            z_series[df[col].notna()] = z_scores

            if direction == 'both':
                mask = (z_series.abs() > z_thresh)
            elif direction == 'high':
                mask = z_series > z_thresh
            elif direction == 'low':
                mask = z_series < -z_thresh
            else:
                raise ValueError("direction must be one of ['both', 'high', 'low']")

        elif method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_mult * IQR
            upper = Q3 + iqr_mult * IQR

            if direction == 'both':
                mask = (df[col] < lower) | (df[col] > upper)
            elif direction == 'high':
                mask = df[col] > upper
            elif direction == 'low':
                mask = df[col] < lower
            else:
                raise ValueError("direction must be one of ['both', 'high', 'low']")
        else:
            raise ValueError("method must be 'zscore' or 'iqr'")

        if strategy == 'remove':
            rows_to_remove |= mask

        elif strategy == 'replace_mean':
            mean_val = df[col][~mask].mean()
            df.loc[mask, col] = mean_val

        elif strategy == 'replace_median':
            median_val = df[col][~mask].median()
            df.loc[mask, col] = median_val

        elif strategy == 'clip':
            if method == 'zscore':
                col_mean = df[col].mean()
                col_std = df[col].std()
                lower = col_mean - z_thresh * col_std
                upper = col_mean + z_thresh * col_std
            df[col] = df[col].clip(lower=lower, upper=upper)

        if verbose:
            print(f"[{col}] 异常值数量: {mask.sum()}，处理方式：{strategy}")

    if strategy == 'remove':
        df = df[~rows_to_remove]

    return df