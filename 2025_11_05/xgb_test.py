import xgboost as xgb
import numpy as np

print(xgb.__version__)  # 2.1.1 나올 것

X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=int)

dtrain = xgb.DMatrix(X, label=y)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",   # 이제는 이거
    "device": "cuda",        # GPU 사용 선언
    # "predictor": "gpu_predictor",  # 생략해도 device=cuda면 자동으로 맞춰 줌
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=50,
    verbose_eval=10
)

print("OK, trained with CUDA device")