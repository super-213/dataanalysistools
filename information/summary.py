import pandas as pd
import io

def summarize_df(df: pd.DataFrame, head_rows: int = 5) -> dict:
    """
    汇总 DataFrame 的基本信息，包括 head, describe, info 等内容。
    
    参数:
        df (pd.DataFrame): 需要汇总的 DataFrame
        head_rows (int): 显示前几行数据,默认5行
    
    返回:
        dict: 包含 head, tail, describe, info, dtypes, missing 的字典结果
    """
    summary = {}
    # 获取 df.head()
    summary['head'] = df.head(head_rows)
    print("-" * 50)
    print("df.head():")
    print(summary['head'])

    # 获取 df.tail()
    summary['tail'] = df.tail(head_rows)
    print("-" * 50)
    print("df.tail():")
    print(summary['tail'])

    # 获取 df.describe()
    summary['describe'] = df.describe(include='all')
    print("-" * 50)
    print("df.describe():")
    print(summary['describe'])

    # 获取 df.info() 的文本结果
    buffer = io.StringIO()
    df.info(buf=buffer)
    summary['info'] = buffer.getvalue()
    print("-" * 50)
    print("df.info():")
    print(summary['info'])

    # 添加列的数据类型
    summary['dtypes'] = df.dtypes
    print("-" * 50)
    print("df.dtypes:")
    print(summary['dtypes'])

    # 添加缺失值统计
    summary['missing'] = df.isnull().sum()
    print("-" * 50)
    print("缺失值统计 df.isnull().sum():")
    print(summary['missing'])

    return summary

def categorical_summary(
    df: pd.DataFrame,
    cols: list = None,
    topn: int = 10,
    dropna: bool = True,
    verbose: bool = True
) -> dict:
    """
    统计每个类别型变量的类别数和频数分布（可选前N个）。

    参数:
        df (pd.DataFrame): 输入 DataFrame。
        cols (list): 指定要分析的列名列表，默认分析所有 object / category 类型列。
        topn (int): 显示每列前 topn 个类别及其频数。
        dropna (bool): 是否忽略 NaN。
        verbose (bool): 是否打印详细结果。

    返回:
        dict: 每列的统计信息，结构为：
            {
                'column1': {
                    'n_unique': int,
                    'value_counts': pd.Series
                },
                ...
            }
    """
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    summary = {}

    for col in cols:
        vc = df[col].value_counts(dropna=dropna)
        summary[col] = {
            'n_unique': df[col].nunique(dropna=dropna),
            'value_counts': vc.head(topn)
        }

        if verbose:
            print(f"\n{'-'*50}")
            print(f"Column: {col}")
            print(f"Unique Categories: {summary[col]['n_unique']}")
            print(f"Top {topn} Category Distribution:")
            print(summary[col]['value_counts'])

    return summary