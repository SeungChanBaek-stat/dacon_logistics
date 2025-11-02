# xgb_focal_A.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score
)
import xgboost as xgb

# --------- 경로 ----------
curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# --------- 데이터 로드 & 병합 ----------
Atrain_processed = pd.read_csv(os.path.join(DATA_DIR, "trainA_processed_fast.csv"))
Atrain_labels    = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

A_labels = Atrain_labels.query("Test == 'A'").copy()
trainA = pd.merge(
    A_labels, Atrain_processed, on="Test_id", how="inner",
    validate='one_to_one', suffixes=('', '_proc')
)
assert (trainA['Test'] == trainA['Test_proc']).all()
trainA = trainA.drop(columns=['Test_proc'])

# --------- X, y 구성 ----------
drop_cols = ['Test_id', 'Test']
y = trainA['Label'].astype(int).values
X = trainA.drop(columns=drop_cols + ['Label'])
# (선택) 안전하게 float32
X = X.apply(pd.to_numeric, errors='coerce').astype(np.float32)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"neg = {(y_tr==0).sum()} pos = {(y_tr==1).sum()}")

# --------- Focal Loss (binary) : grad/hess 커스텀 ----------
# FL = - alpha * y * (1-p)^gamma * log(p) - (1-alpha) * (1-y) * p^gamma * log(1-p)
# 여기선 실무에서 검증된 '안정적 근사' 공식을 사용 (수치안정 + 속도 균형)
def focal_binary_objective(alpha=0.25, gamma=2.0, eps=1e-12):
    def _obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        # raw margin -> probability
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, eps, 1.0 - eps)

        # class-wise modulating factor
        # w1 = alpha * (1-p)^gamma   for y=1
        # w0 = (1-alpha) * p^gamma   for y=0
        w1 = alpha * np.power(1.0 - p, gamma)
        w0 = (1.0 - alpha) * np.power(p, gamma)
        w  = y * w1 + (1.0 - y) * w0

        # base grad/hess of BCE w.r.t. margin z : grad = p - y, hess = p*(1-p)
        g_bce = (p - y)
        h_bce = p * (1.0 - p)

        # 추가 항: focal modulation의 도함수 근사 (안정적 실전 버전)
        # mod_grad ~ w * (1 + gamma * (y*( -np.log(p)) + (1-y)*( -np.log(1-p) )))
        mod_term = (y * (-np.log(p)) + (1.0 - y) * (-np.log(1.0 - p)))
        g = w * g_bce * (1.0 + gamma * mod_term)

        # hessian은 과대 추정 방지 위해 보수적 근사(안정성 우선)
        # h ≈ w * h_bce * (1.0 + gamma * mod_term)
        h = w * h_bce * (1.0 + gamma * mod_term)

        return g, h
    return _obj

def focal_binary_metric(alpha=0.25, gamma=2.0, eps=1e-12):
    def _metric(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, eps, 1.0 - eps)
        fl = - alpha * y * np.power(1.0 - p, gamma) * np.log(p) \
             - (1.0 - alpha) * (1.0 - y) * np.power(p, gamma) * np.log(1.0 - p)
        return 'focal_loss', float(np.mean(fl))   # ← 2-튜플만 반환
    return _metric

# --------- XGBoost 세팅 ----------
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dvalid = xgb.DMatrix(X_va, label=y_va)

# 클래스 비율 참고
neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
scale_pos_weight = max(1.0, neg / max(pos, 1))

params = {
    "max_depth": 7,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "objective": "binary:logistic",  # placeholder (custom obj 사용)
    "eval_metric": "auc",            # AUC도 함께 트래킹
    "tree_method": "hist",           # CPU 고속 학습
    "scale_pos_weight": scale_pos_weight * 0.5,  # focal과 함께면 과하게 주지 않음
    "verbosity": 1,
}

alpha, gamma = 0.25, 2.0
obj = focal_binary_objective(alpha=alpha, gamma=gamma)
metric = focal_binary_metric(alpha=alpha, gamma=gamma)

evallist = [(dtrain, "train"), (dvalid, "valid")]
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=6000,
    evals=evallist,
    obj=obj,
    custom_metric=metric,
    early_stopping_rounds=300,
    verbose_eval=100
)

# --------- 검증 평가지표 & 임계값 튜닝 ----------
p_valid = bst.predict(dvalid, iteration_range=(0, bst.best_iteration+1))
roc = roc_auc_score(y_va, p_valid)
prec, rec, thr = precision_recall_curve(y_va, p_valid)
ap = average_precision_score(y_va, p_valid)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = f1s.argmax()
best_thr = thr[max(best_idx-1, 0)] if best_idx == len(thr) else thr[best_idx]

print(f"[VAL] ROC-AUC={roc:.4f}  PR-AUC(AP)={ap:.4f}  bestF1={f1s[best_idx]:.4f} @thr={best_thr:.4f}")

# --------- 중요도/저장 ----------
imp = bst.get_score(importance_type='gain')
imp = pd.Series(imp).sort_values(ascending=False)
print("\n[Top-20 Feature Importances by gain]")
print(imp.head(20))

bst.save_model(os.path.join(OUT_DIR, "xgb_focal_A.json"))
np.save(os.path.join(OUT_DIR, "xgb_focal_A_features.npy"), X.columns.values)
print(f"[OK] Saved model + feature names to {OUT_DIR}")
