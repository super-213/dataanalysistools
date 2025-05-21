```bash
import mytools as mt
clean_df = mt.clean.impute(raw_df, strategy="median")
```

```bash
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["A", "B", "C"],
    "score": [87.5, 90.0, 76.5],
    "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
})

# 按列名删除
df_clean = drop_columns(df, cols_to_drop=["id", "timestamp"])

# 按数据类型删除
df_clean = drop_columns(df, dtypes_to_drop=["object"])

# 按列名和数据类型删除
df_clean = drop_columns(df,
                        cols_to_drop=["score"],
                        dtypes_to_drop=["datetime64[ns]"],
                        verbose=True)
```

```bash
# 示例数据
df = pd.DataFrame({
    "A": [10, 12, 15, 14, 13, 100],    # 100 是异常
    "B": [1, 1.1, 0.9, 1.2, 1.05, 0.95]
})

# 仅检测高值异常 + 返回 Z 分数
outliers = detect_outliers_zscore(
    df, cols=["A"],
    threshold=2.5,
    method="high",
    return_zscore=True
)
print(outliers)

# 直接剔除异常值
df_clean = detect_outliers_zscore(df, threshold=3.0, drop=True)
```

```bash
# 示例数据
df = pd.DataFrame({
    "A": [10, 12, 15, 14, 13, 100],
    "B": [1, 1.1, 0.9, 1.2, 1.05, 0.95]
})

# 检测异常值
outliers = detect_outliers_iqr(df, method="both", multiplier=1.5, verbose=True)

# 剔除异常值
df_clean = detect_outliers_iqr(df, drop=True)

# 检测并获取上下界
outliers, bounds = detect_outliers_iqr(df, return_bounds=True)
print(bounds)
```

```bash
# 示例 DataFrame
df = pd.DataFrame({
    'A': [10, 12, 15, 14, 13, 100],
    'B': [1, 1.1, 0.9, 1.2, 1.05, 0.95]
})

# 使用 Z-Score 删除异常值
df_clean = handle_outliers(df, method='zscore', strategy='remove', verbose=True)

# 使用 IQR 将异常值替换为中位数
df_replaced = handle_outliers(df, method='iqr', strategy='replace_median', verbose=True)

# 限制异常值在范围内（Z-Score）
df_clipped = handle_outliers(df, method='zscore', strategy='clip', z_thresh=2.5, verbose=True)
```