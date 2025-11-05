import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, precision_recall_curve, average_precision_score

from sklearn.impute import SimpleImputer
import lightgbm as lgb
import joblib

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
Atrain_processed = os.path.join(DATA_DIR, "A_processed", "trainA_processed_fast.csv")
Atrain_labels = os.path.join(DATA_DIR, "train.csv")

Atrain_processed = pd.read_csv(Atrain_processed)
Atrain_labels = pd.read_csv(Atrain_labels)

A_labels = Atrain_labels.query("Test == 'A'").copy()

trainA = pd.merge(
    A_labels, Atrain_processed,
    on='Test_id', how='inner',
    validate='one_to_one', suffixes=('', '_proc')
)

# 두 Test가 동일한지 확인 후 하나만 남기기
assert (trainA['Test'] == trainA['Test_proc']).all()
trainA = trainA.drop(columns=['Test_proc'])


# --- 1) 불필요한 칼럼 제거 & X, y 분리 ---
drop_cols = ['Test_id', 'Test']  # 모델에 불필요
assert set(drop_cols + ['Label']).issubset(trainA.columns), "필수 칼럼 누락 확인"

y_trainA = trainA['Label'].astype(int).values
X_trainA = trainA.drop(columns=drop_cols + ['Label'])

# # 수치형으로 강제 변환(혹시 모를 문자/혼합형 방지), 불가 항목은 NaN
# X_trainA = X_trainA.apply(pd.to_numeric, errors='coerce').astype(np.float32)

# --- 3) 학습/검증 분할 (stratify 권장) ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainA, y_trainA, test_size=0.05, random_state=42, stratify=y_trainA
)






# --- 4) LightGBM 모델 학습 ---
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / max(pos, 1)
print(f"neg = {neg}", f"pos = {pos}")





# clf = lgb.LGBMClassifier(
#     objective="binary",
#     n_estimators=8000,
#     learning_rate=0.02,
#     num_leaves=31,
#     max_depth=7,
#     min_data_in_leaf=100,
#     feature_fraction=0.8,
#     bagging_fraction=0.8,
#     bagging_freq=5,
#     scale_pos_weight=spw,         # ★ 불균형 보정
#     force_col_wise=True,          # 로그에서 제안된 최적화
#     n_jobs=-1,
#     # verbosity는 경고/정보 레벨용(훈련 로그는 callbacks로)
#     verbosity=1,
# )
# clf.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     eval_metric="auc",
#     callbacks=[
#         lgb.early_stopping(stopping_rounds=300),
#         lgb.log_evaluation(period=100),
#     ],
# )

# # --- 5) 평가 ---
# # proba_va = clf.predict_proba(X_valid)[:, 1]
# # pred_va  = (proba_va >= 0.5).astype(int)

# # auc = roc_auc_score(y_valid, proba_va)
# # f1  = f1_score(y_valid, pred_va)
# # acc = accuracy_score(y_valid, pred_va)
# # print(f"[VAL] AUC={auc:.4f}  F1={f1:.4f}  ACC={acc:.4f}")
# # print(classification_report(y_valid, pred_va, digits=4))

# # # --- 6) 피처 중요도 (상위 20개) ---
# # feat_imp = pd.Series(clf.feature_importances_, index=X_trainA.columns).sort_values(ascending=False)
# # print("\n[Top-20 Feature Importances]")
# # print(feat_imp.head(20))

# # # --- 7) 모델/전처리 저장 ---
# # os.makedirs(OUT_DIR, exist_ok=True)
# # joblib.dump({'model': clf, 'feature_names': X_trainA.columns.tolist()},
# #             os.path.join(OUT_DIR, 'lgbm_A_model.pkl'))
# # print(f"[OK] Saved model to {os.path.join(OUT_DIR, 'lgbm_A_model.pkl')}")

# # --- 검증 성능 및 임계값 튜닝
# p_valid = clf.predict_proba(X_valid)[:,1]
# roc = roc_auc_score(y_valid, p_valid)
# prec, rec, thr = precision_recall_curve(y_valid, p_valid)
# f1s = 2 * prec * rec / (prec + rec + 1e-12)
# best_idx = f1s.argmax()
# best_thr = thr[max(best_idx, 0)]
# ap = average_precision_score(y_valid, p_valid)
# y_hat = (p_valid >= best_thr).astype(int)

# print(f"[VAL] ROC-AUC={roc:.4f}  PR-AUC(AP)={ap:.4f}  bestF1={f1s[best_idx]:.4f} @thr={best_thr:.4f}")