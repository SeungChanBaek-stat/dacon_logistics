import os
import sys
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

import lightgbm as lgb

# ===========================
# 대회용 스코어 함수들
# ===========================
from sklearn.metrics import mean_squared_error, roc_auc_score
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from functions.Validation import auc_brier_ece



# ===========================
# LightGBM 10-fold CV 함수
# ===========================
def run_lgb_10fold(
    X: pd.DataFrame,
    y: np.ndarray,
    test_ids: np.ndarray,
    test_name: str = "A",
    n_splits: int = 10,
    random_state: int = 42,
):
    """
    X, y, Test_id를 받아서 LightGBM 10-fold CV 수행.
    - OOF 확률 반환
    - 전체 ROC-AUC, PR-AUC(AP), best F1, auc_brier_ece 출력
    """

    X = X.reset_index(drop=True)
    y = np.asarray(y)
    test_ids = np.asarray(test_ids)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    oof_pred = np.zeros(len(y), dtype=float)
    fold_metrics = []

    print(f"\n===== [{test_name}] LightGBM {n_splits}-Fold CV 시작 =====")
    print(f"데이터 크기: {X.shape},  양성 비율: {y.mean():.4f}")

    # best λ₁=0.1, λ₂=0
    l1, l2 = 0.1, 0
    n_jobs = 8
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        

        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_valid = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        spw = neg / max(pos, 1)
        print(f"\n[Fold {fold}] neg={neg}, pos={pos}, scale_pos_weight={spw:.2f}")

        params = {
            "objective": "binary",
            "metric": "auc",         # 모니터링용
            "learning_rate": 0.02,
            "num_leaves": 32,
            "min_data_in_leaf": 200, # ← 여기 추가
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
            "verbosity": -1,
            "lambda_l1": l1,
            "lambda_l2": l2,            
            "n_jobs": n_jobs
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            # **lgb.callback = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]**
            # , lgb.early_stopping(stopping_rounds=100)
            callbacks = [lgb.log_evaluation(20), lgb.early_stopping(stopping_rounds=100)]
            # early_stopping_rounds=???  # 쓸지 말지는 우리가 결정
        )


        # 4) 검증셋 확률 예측
        p_va = model.predict(X_va, num_iteration=model.best_iteration)  # 0~1 확률
        
        # 5) 대회 메트릭 계산
        answer_df = pd.DataFrame({
            "id": np.arange(len(y_va)),
            "Label": y_va.astype(int)
        })
        
        submission_df = pd.DataFrame({
            "id": np.arange(len(p_va)),
            "Label": p_va.astype(float)
        })
        
        fold_score = auc_brier_ece(answer_df, submission_df)
        print(f"Fold {fold} combined score: {fold_score:.6f}, elapsed : {time.time()-t0:.3f}s")
        fold_scores.append(fold_score)

        


        oof_pred[va_idx] = p_va

        # fold별 지표 계산
        roc = roc_auc_score(y_va, p_va)
        prec, rec, thr = precision_recall_curve(y_va, p_va)
        ap = average_precision_score(y_va, p_va)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = f1s.argmax()
        best_thr = thr[max(best_idx - 1, 0)] if best_idx == len(thr) else thr[best_idx]
        best_f1 = f1s[best_idx]

        fold_metrics.append((roc, ap, best_f1))

        print(
            f"[Fold {fold} 결과] ROC-AUC={roc:.4f}  PR-AUC(AP)={ap:.4f}  "
            f"bestF1={best_f1:.4f} @thr={best_thr:.4f}"
            f"auc_brier_ece = {fold_score:.6f}"
        )



    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    # ===== 전체 OOF 기준 최종 지표 =====
    print(f"\n===== [{test_name}] 전체 OOF 기준 최종 성능 =====")
    roc_all = roc_auc_score(y, oof_pred)
    prec_all, rec_all, thr_all = precision_recall_curve(y, oof_pred)
    ap_all = average_precision_score(y, oof_pred)
    f1s_all = 2 * prec_all * rec_all / (prec_all + rec_all + 1e-12)
    best_idx_all = f1s_all.argmax()
    best_thr_all = thr_all[max(best_idx_all - 1, 0)] if best_idx_all == len(thr_all) else thr_all[best_idx_all]
    best_f1_all = f1s_all[best_idx_all]

    print(f"[OOF] ROC-AUC={roc_all:.4f}  PR-AUC(AP)={ap_all:.4f}  "
          f"bestF1={best_f1_all:.4f} @thr={best_thr_all:.4f}")


    print(f"[OOF] auc_brier_ece 평균 = {mean_score:.6f}, 분산 = {std_score:.6f}  (낮을수록 좋음)")

    # fold별 평균 지표도 같이 보여주기
    fold_roc = np.mean([m[0] for m in fold_metrics])
    fold_ap = np.mean([m[1] for m in fold_metrics])
    fold_f1 = np.mean([m[2] for m in fold_metrics])
    print(f"[Fold 평균] ROC-AUC={fold_roc:.4f}  PR-AUC(AP)={fold_ap:.4f}  F1={fold_f1:.4f}")


    return oof_pred


# ===========================
# 메인: 데이터 로드하고 A/B 각각 CV 돌리기
# ===========================
if __name__ == "__main__":
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    DATA_DIR = os.path.join(parent_path, "data")
    A_processed_dir = os.path.join(DATA_DIR, "A_processed")
    B_processed_dir = os.path.join(DATA_DIR, "B_processed")

    labels_path = os.path.join(DATA_DIR, "train.csv")
    trainA_processed_path = os.path.join(A_processed_dir, "trainA_processed_fast.csv")
    trainB_processed_path = os.path.join(B_processed_dir, "trainB_processed_fast.csv")

    trainA_processed = pd.read_csv(trainA_processed_path)
    trainB_processed = pd.read_csv(trainB_processed_path)
    labels = pd.read_csv(labels_path)

    # A, B 라벨 분리
    A_labels = labels.query("Test == 'A'").copy()
    B_labels = labels.query("Test == 'B'").copy()

    # merge
    trainA = pd.merge(
        A_labels,
        trainA_processed,
        on="Test_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_proc"),
    )
    trainB = pd.merge(
        B_labels,
        trainB_processed,
        on="Test_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_proc"),
    )

    # Test 값 일치 확인 후 하나만 사용
    assert (trainA["Test"] == trainA["Test_proc"]).all()
    assert (trainB["Test"] == trainB["Test_proc"]).all()
    trainA = trainA.drop(columns=["Test_proc"])
    trainB = trainB.drop(columns=["Test_proc"])

    # 기본 정보 확인
    print("[trainA] shape:", trainA.shape)
    print("[trainB] shape:", trainB.shape)
    print("[Label 분포 A]")
    print(trainA["Label"].value_counts())
    print("[Label 분포 B]")
    print(trainB["Label"].value_counts())

    # X, y 분리
    drop_cols = ["Test_id", "Test", "Label"]

    yA = trainA["Label"].astype(int).values
    XA = trainA.drop(columns=drop_cols)
    idsA = trainA["Test_id"].values

    yB = trainB["Label"].astype(int).values
    XB = trainB.drop(columns=drop_cols)
    idsB = trainB["Test_id"].values

    # 필요시 안전하게 수치형 변환
    XA = XA.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    XB = XB.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # === A, B 각각 10-fold CV 실행 ===
    oof_A = run_lgb_10fold(XA, yA, idsA, test_name="A", n_splits=10)
    oof_B = run_lgb_10fold(XB, yB, idsB, test_name="B", n_splits=10)