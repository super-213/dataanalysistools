from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import numpy as np

def bayes_optimize_model(
    model_class,
    param_bounds: dict,
    X, y,
    cv: int = 5,
    scoring: str = 'accuracy',
    init_points: int = 5,
    n_iter: int = 20,
    verbose: bool = True,
    fixed_params: dict = None
):
    """
    使用贝叶斯优化寻找最佳模型参数。

    参数：
    - model_class：模型类（例如 XGBClassifier、RandomForestClassifier）
    - param_bounds：超参数搜索空间（例如 {'max_depth': (3, 10), ...}）
    - X, y：训练数据
    - cv：交叉验证折数
    - scoring：评估指标（如 'accuracy'）
    - init_points：初始随机搜索次数
    - n_iter：迭代次数
    - verbose：是否输出日志
    - fixed_params：固定参数（如 XGB 的 use_label_encoder=False）

    返回：
    - optimizer：BayesianOptimization 对象（可提取最佳参数）
    """
    if fixed_params is None:
        fixed_params = {}

    def objective_function(**params):
        # 将浮点参数转为整数（如果是整数类型）
        for key, value in params.items():
            if isinstance(param_bounds[key][0], int):
                params[key] = int(round(value))
        model = model_class(**params, **fixed_params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        if verbose:
            print(f"Params: {params} | Score: {scores.mean():.4f}")
        return scores.mean()

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer