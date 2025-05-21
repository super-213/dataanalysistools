```bash
df = pd.DataFrame({
    'height': [160, 170, 180, 190],
    'weight': [55, 65, 75, 85]
})

# 标准化所有数值列，使用 Z-Score 方法
standardized_df = standardize_data(df, method='zscore')

# 使用 Min-Max 标准化并获取 scaler
standardized_df, scaler = standardize_data(df, method='minmax', return_scaler=True)

# 使用 Robust 标准化并原地修改
standardize_data(df, method='robust', inplace=True)
```

```bash
import pandas as pd
from data_analysis_tools.features.binning import bin_continuous_variable

df = pd.DataFrame({
    'age': [18, 22, 25, 28, 30, 35, 40, 45, 50, 60]
})

# 等宽分箱（5段）
df1 = bin_continuous_variable(df, 'age', method='equal_width', bins=5)

# 等频分箱（4组）
df2 = bin_continuous_variable(df, 'age', method='equal_freq', bins=4)

# 自定义边界
df3 = bin_continuous_variable(df, 'age', method='custom', bins=[0, 25, 40, 60], labels=['青年', '中年', '老年'])
```