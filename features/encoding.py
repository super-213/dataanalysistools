import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def encode_columns(df: pd.DataFrame,
                   columns: list,
                   method: str = 'label',
                   drop_original: bool = True,
                   **kwargs) -> pd.DataFrame:
    """
    对 DataFrame 中的指定列进行编码

    参数：
        df (pd.DataFrame): 要处理的数据
        columns (list): 需要编码的列名
        method (str): 编码方法，支持 'label', 'onehot', 'ordinal'
        drop_original (bool): 是否删除原始列（对于 onehot 有效）
        kwargs: 传给编码器的其他参数

    返回：
        pd.DataFrame: 编码后的新 DataFrame
    """
    df = df.copy()

    if method == 'label':
        for col in columns:
            encoder = LabelEncoder()
            df[col + '_label'] = encoder.fit_transform(df[col].astype(str))
            if drop_original:
                df.drop(columns=col, inplace=True)

    elif method == 'onehot':
        encoder = OneHotEncoder(sparse=False, dtype=int, **kwargs)
        transformed = encoder.fit_transform(df[columns])
        new_cols = encoder.get_feature_names_out(columns)
        df_onehot = pd.DataFrame(transformed, columns=new_cols, index=df.index)
        df = pd.concat([df, df_onehot], axis=1)
        if drop_original:
            df.drop(columns=columns, inplace=True)

    elif method == 'ordinal':
        encoder = OrdinalEncoder(**kwargs)
        df_ordinal = encoder.fit_transform(df[columns])
        for i, col in enumerate(columns):
            df[col + '_ordinal'] = df_ordinal[:, i]
        if drop_original:
            df.drop(columns=columns, inplace=True)

    else:
        raise ValueError(f"不支持的编码方法：{method}，请选择 'label', 'onehot', 或 'ordinal'")

    return df