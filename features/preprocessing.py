import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import boxcox, yeojohnson

def standardize_data(
    df: pd.DataFrame,
    cols: list = None,
    method: str = 'zscore',  # 'zscore', 'minmax', 'robust', 'log', 'boxcox', 'yeojohnson', 'maxabs', 'l2'
    return_scaler: bool = False,
    inplace: bool = False
):
    """
    对 DataFrame 中的列进行标准化处理。

    参数:
    - df : pandas.DataFrame
    - cols : 需要标准化的列名列表，默认所有数值列
    - method : 标准化方法 ['zscore', 'minmax', 'robust', 
    'log', 'boxcox', 'yeojohnson', 'maxabs', 'l2']
    - return_scaler : 是否返回用于反标准化的参数
    - inplace : 是否在原始 DataFrame 上修改

    返回:
    - df_new: 标准化后的 DataFrame（或原始 DataFrame）
    - scaler: 包含标准化参数（可选）
    """
    df_new = df if inplace else df.copy()
    if cols is None:
        cols = df_new.select_dtypes(include=[np.number]).columns.tolist()

    scaler = {}

    for col in cols:
        series = df_new[col]

        if method == 'zscore':
            mean = series.mean()
            std = series.std()
            df_new[col] = (series - mean) / std
            scaler[col] = {'mean': mean, 'std': std}

        elif method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            df_new[col] = (series - min_val) / (max_val - min_val)
            scaler[col] = {'min': min_val, 'max': max_val}
        
        elif method == 'log':
            if (series <= 0).any():
                raise ValueError(f"Column {col} contains non-positive values, cannot apply log transformation.")
            df_new[col] = np.log1p(series)
            scaler[col] = {'method': 'log1p'}
        
        elif method == 'boxcox':
            shifted = series - series.min() + 1
            df_new[col], lambda_val = boxcox(shifted)
            scaler[col] = {'lambda': lambda_val, 'shift': float(series.min()) - 1}

        elif method == 'yeojohnson':
            df_new[col], lambda_val = yeojohnson(series)
            scaler[col] = {'lambda': lambda_val}

        elif method == 'maxabs':
            max_abs = series.abs().max()
            if max_abs == 0:
                df_new[col] = 0  # 避免除以0
                scaler[col] = {'max_abs': 0}
            else:
                df_new[col] = series / max_abs
                scaler[col] = {'max_abs': max_abs}
        
        elif method == 'l2':
            norm = np.linalg.norm(series, ord=2)
            if norm == 0:
                df_new[col] = 0
                scaler[col] = {'norm': 0}
            else:
                df_new[col] = series / norm
                scaler[col] = {'norm': norm}

        elif method == 'robust':
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            df_new[col] = (series - median) / iqr
            scaler[col] = {'median': median, 'iqr': iqr}

        else:
            raise ValueError("method must be one of ['zscore', 'minmax', 'robust', 'log', 'boxcox', 'yeojohnson', 'maxabs', 'l2']")

    if return_scaler:
        return df_new, scaler
    else:
        return df_new

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def normalize_data(
    df: pd.DataFrame,
    cols: list = None,
    method: str = 'minmax',  # ['minmax', 'maxabs', 'log']
    feature_range: tuple = (0, 1),  # 仅对 minmax 有效
    return_params: bool = False,
    inplace: bool = False
):
    """
    对 DataFrame 中的指定列进行归一化处理。

    参数:
    - df : pandas.DataFrame
    - cols : 需要归一化的列名列表，默认所有数值列
    - method : 归一化方法 ['minmax', 'maxabs', 'log']
    - feature_range : 特征缩放范围，仅对 minmax 有效，默认 (0, 1)
    - return_params : 是否返回归一化参数，用于反归一化
    - inplace : 是否在原始 DataFrame 上修改

    返回:
    - df_new : 归一化后的 DataFrame
    - params : 包含归一化参数（可选）
    """
    df_new = df if inplace else df.copy()
    if cols is None:
        cols = df_new.select_dtypes(include=[np.number]).columns.tolist()

    params = {}
    range_min, range_max = feature_range

    for col in cols:
        series = df_new[col]

        if method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                df_new[col] = 0  # 避免除以0
                params[col] = {'min': min_val, 'max': max_val}
                continue
            scaled = (series - min_val) / (max_val - min_val)
            df_new[col] = scaled * (range_max - range_min) + range_min
            params[col] = {'min': min_val, 'max': max_val, 'range': feature_range}

        elif method == 'maxabs':
            max_abs = series.abs().max()
            df_new[col] = series / max_abs
            params[col] = {'max_abs': max_abs}

        elif method == 'log':
            df_new[col] = np.log1p(series)
            params[col] = {'method': 'log1p'}

        else:
            raise ValueError("method must be one of ['minmax', 'maxabs', 'log']")

    if return_params:
        return df_new, params
    else:
        return df_new


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


def detect_skewness(
    df: pd.DataFrame,
    threshold: float = 1.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    检测数值型列的偏度，并根据偏度值推荐处理方式。

    参数:
    - df: pandas DataFrame
    - threshold: 偏度绝对值超过该值时认为有显著偏态（默认=1.0）
    - verbose: 是否打印推荐建议（可选）

    返回:
    - result_df: 包含列名、偏度值、偏态类型、推荐处理方法的 DataFrame
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    results = []

    for col in numeric_cols:
        skew_val = df[col].skew()
        if np.isnan(skew_val):
            continue

        if abs(skew_val) < 0.5:
            skew_type = '近似正态'
            suggestion = '无需处理'
        elif 0.5 <= skew_val < threshold:
            skew_type = '中度右偏'
            suggestion = '可尝试 log(x+1), sqrt(x)'
        elif -threshold < skew_val <= -0.5:
            skew_type = '中度左偏'
            suggestion = '可尝试 log(max+1 - x), sqrt(max - x)'
        elif skew_val >= threshold:
            skew_type = '严重右偏'
            suggestion = '建议尝试 log(x+1), box-cox'
        else:
            skew_type = '严重左偏'
            suggestion = '建议尝试反转数据再 log/box-cox'

        results.append({
            'Column': col,
            'Skewness': round(skew_val, 3),
            'Type': skew_type,
            'Suggested_Transform': suggestion
        })

        if verbose:
            print(f"{col}: Skew = {skew_val:.3f}, 类型 = {skew_type}, 推荐 = {suggestion}")

    result_df = pd.DataFrame(results)
    return result_df

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson

class DataStandardizer:
    def __init__(self, method='zscore', cols=None):
        """
        初始化标准化器。

        参数:
        - method: 选择的标准化方法 ['zscore', 'minmax', 'robust', 'log',
                  'boxcox', 'yeojohnson', 'maxabs', 'l2']
        - cols: 需要标准化的列名列表，默认为 None（表示所有数值列）
        """
        self.method = method
        self.cols = cols
        self.scaler_params = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame, cols: list = None):
        """
        拟合标准化参数。

        参数:
        - df: 输入 DataFrame
        - cols: 指定的列，默认为所有数值列
        """
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cols = cols
        self.scaler_params = {}

        for col in cols:
            series = df[col]

            if self.method == 'zscore':
                self.scaler_params[col] = {
                    'mean': series.mean(),
                    'std': series.std()
                }

            elif self.method == 'minmax':
                self.scaler_params[col] = {
                    'min': series.min(),
                    'max': series.max()
                }

            elif self.method == 'log':
                if (series <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive values, cannot apply log transformation.")
                self.scaler_params[col] = {'method': 'log1p'}

            elif self.method == 'boxcox':
                shift = series.min() - 1
                shifted = series - shift
                _, lmbda = boxcox(shifted)
                self.scaler_params[col] = {'lambda': lmbda, 'shift': shift}

            elif self.method == 'yeojohnson':
                _, lmbda = yeojohnson(series)
                self.scaler_params[col] = {'lambda': lmbda}

            elif self.method == 'maxabs':
                max_abs = series.abs().max()
                self.scaler_params[col] = {'max_abs': max_abs}

            elif self.method == 'l2':
                norm = np.linalg.norm(series, ord=2)
                self.scaler_params[col] = {'norm': norm}

            elif self.method == 'robust':
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                self.scaler_params[col] = {
                    'median': series.median(),
                    'iqr': q3 - q1
                }

            else:
                raise ValueError(f"Unsupported method: {self.method}")

        self.fitted = True

    def transform(self, df: pd.DataFrame, inplace=False):
        """
        使用拟合的参数对数据进行标准化。

        参数:
        - df: 输入 DataFrame
        - inplace: 是否在原始 DataFrame 上修改

        返回:
        - 标准化后的 DataFrame
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")

        df_new = df if inplace else df.copy()

        for col in self.cols:
            series = df[col]
            param = self.scaler_params[col]

            if self.method == 'zscore':
                df_new[col] = (series - param['mean']) / param['std']

            elif self.method == 'minmax':
                df_new[col] = (series - param['min']) / (param['max'] - param['min'])

            elif self.method == 'log':
                df_new[col] = np.log1p(series)

            elif self.method == 'boxcox':
                shifted = series - param['shift']
                df_new[col] = boxcox(shifted, lmbda=param['lambda'])

            elif self.method == 'yeojohnson':
                df_new[col] = yeojohnson(series, lmbda=param['lambda'])

            elif self.method == 'maxabs':
                if param['max_abs'] == 0:
                    df_new[col] = 0
                else:
                    df_new[col] = series / param['max_abs']

            elif self.method == 'l2':
                if param['norm'] == 0:
                    df_new[col] = 0
                else:
                    df_new[col] = series / param['norm']

            elif self.method == 'robust':
                df_new[col] = (series - param['median']) / param['iqr']

        return df_new

    def fit_transform(self, df: pd.DataFrame, cols: list = None, inplace=False):
        """
        拟合并标准化数据。
        """
        self.fit(df, cols)
        return self.transform(df, inplace=inplace)