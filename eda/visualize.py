import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import math



def pairwise_plot(
    df: pd.DataFrame,
    features: list,
    plot_type: str = 'scatter',  # 可选: 'scatter', 'box', 'bar', 'violin', 'hist'
    hue: str = None,
    max_per_figure: int = 9,
    figsize: tuple = (15, 12),
    save: bool = False,
    save_prefix: str = "pairplot"
):
    """
    批量绘制特征两两组合的图表，每张大图包含最多9张子图。

    参数:
    - df: DataFrame 数据
    - features: 要组合的特征列名列表
    - plot_type: 图表类型：'scatter', 'box', 'violin', 'hist'
    - hue: 分类变量（可选）
    - max_per_figure: 每页最多显示几个子图
    - figsize: 每张图的整体大小
    - save: 是否保存图像
    - save_prefix: 图像保存的前缀
    """
    combs = list(itertools.combinations(features, 2))
    total_figures = math.ceil(len(combs) / max_per_figure)

    for fig_idx in range(total_figures):
        fig, axs = plt.subplots(3, 3, figsize=figsize)
        axs = axs.flatten()
        start = fig_idx * max_per_figure
        end = start + max_per_figure
        current_combs = combs[start:end]

        for ax_idx, (x, y) in enumerate(current_combs):
            ax = axs[ax_idx]
            if plot_type == 'scatter':
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
            elif plot_type == 'box':
                sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
            elif plot_type == 'violin':
                sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax)
            elif plot_type == 'bar':
                sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
            elif plot_type == 'hist':
                sns.histplot(data=df, x=x, hue=hue, ax=ax, kde=True)
            else:
                ax.set_title(f"Unsupported plot type: {plot_type}")
            ax.set_title(f"{y} vs {x}")

        # 隐藏多余的子图
        for j in range(len(current_combs), max_per_figure):
            fig.delaxes(axs[j])

        plt.tight_layout()
        if save:
            plt.savefig(f"{save_prefix}_{fig_idx+1}.png", dpi=300)
        plt.show()
    


def plot_all_barplots(df, hue='Class variable', n_cols=3):
    """
    将 DataFrame 中所有列（除 hue）按类别分组绘制为子图中的柱状图

    参数:
    - df: pandas DataFrame，包含数据和类别列
    - hue: str，表示分类的列名
    - n_cols: int，每行显示多少个图

    返回:
    - 无（直接显示整张图）
    """
    # 获取需要绘图的列
    columns = [col for col in df.columns if col != hue]

    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(columns):
        sns.barplot(y=col, data=df, hue=hue, ax=axes[idx])
        axes[idx].set_title(col)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(col)

    # 删除多余子图
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def plot_feature_distributions(
    df: pd.DataFrame,
    features: list = None,
    hue: str = None,
    bins: int = 30,
    kde: bool = True,
    max_per_figure: int = 9,
    figsize: tuple = (15, 12),
    save: bool = False,
    save_prefix: str = "feature_distribution"
):
    """
    批量绘制特征分布图（直方图 + KDE）。

    参数：
    - df: pandas DataFrame
    - features: 要查看分布的列（默认选择数值型列）
    - hue: 分组变量（分类列名）
    - bins: 直方图的 bin 数量
    - kde: 是否绘制 KDE 曲线
    - max_per_figure: 每页显示的图表数
    - figsize: 每页图像大小
    - save: 是否保存图像
    - save_prefix: 保存图像的文件名前缀
    """
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()

    total = len(features)
    total_figures = math.ceil(total / max_per_figure)

    for fig_idx in range(total_figures):
        fig, axs = plt.subplots(3, 3, figsize=figsize)
        axs = axs.flatten()
        start = fig_idx * max_per_figure
        end = start + max_per_figure
        current_features = features[start:end]

        for i, feature in enumerate(current_features):
            ax = axs[i]
            sns.histplot(data=df, x=feature, bins=bins, kde=kde, hue=hue, ax=ax)
            ax.set_title(f"Distribution of {feature}")

        # 删除多余子图
        for j in range(len(current_features), max_per_figure):
            fig.delaxes(axs[j])

        plt.tight_layout()
        if save:
            plt.savefig(f"{save_prefix}_{fig_idx+1}.png", dpi=300)
        plt.show()




def plot_all_boxplots(df, n_cols=3, exclude_columns=None):
    """
    将 DataFrame 中所有数值列绘制为子图中的箱线图

    参数:
    - df: pandas DataFrame
    - n_cols: 每行显示的图数量
    - exclude_columns: 要排除的列名列表，例如类别列

    返回:
    - 无（直接显示整张图）
    """
    if exclude_columns is None:
        exclude_columns = []

    # 只保留数值型列且不在排除列表中
    numeric_columns = df.select_dtypes(include=['number']).columns
    columns = [col for col in numeric_columns if col not in exclude_columns]

    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[idx])
        axes[idx].set_title(f'Boxplot of {col}')
        axes[idx].set_xlabel(col)

    # 删除多余子图
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()