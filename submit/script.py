# run_a_prep_prod.py
import os, sys, time
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 프로젝트 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model.Preprocess import PreprocessA, PreprocessB, parse_age_to_midpoint, split_by_index
from model.Preprocess import _run_block_A, run_parallel_features_A, concat_A, _run_block_B, run_parallel_features_B, concat_B





# ---------- main ----------
def main():
    n_jobs = 3

    # TEST_DIR = "./data"
    curr_path  = os.getcwd()
    # parent_path = os.path.dirname(curr_path)
    print(curr_path)
    # print(parent_path)
    # DATA_DIR = os.path.join(parent_path, "data")
    # train_path = os.path.join(DATA_DIR, "train")
    # trainA_csv = os.path.join(train_path, "A.csv")

    TEST_DIR  = os.path.join(curr_path, "data")             # test.csv, A.csv, B.csv, sample_submission.csv 위치
    test_DIR = os.path.join(TEST_DIR, "test")
    MODEL_DIR = os.path.join(curr_path, "model")             # lgbm_A.pkl, lgbm_B.pkl 위치
    OUT_DIR   = os.path.join(curr_path, "output") 
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    testA_dir = os.path.join(test_DIR, "A.csv")
    testB_dir = os.path.join(test_DIR, "B.csv")
    lgbm_A_pkl_path = os.path.join(MODEL_DIR, "lgbm_A.pkl")
    lgbm_A = joblib.load(lgbm_A_pkl_path)
    xgb_A_json = os.path.join(MODEL_DIR, "xgb_A.json")
    lgbm_B_pkl_path = os.path.join(MODEL_DIR, "lgbm_B.pkl")
    lgbm_B = joblib.load(lgbm_B_pkl_path)
    xgb_B_json = os.path.join(MODEL_DIR, "xgb_B.json")
    OUT_PATH  = os.path.join(OUT_DIR, "submission.csv")




    print("Load XGBoost models ...")
    bst_A = xgb.Booster()
    bst_A.load_model(xgb_A_json)
    bst_B = xgb.Booster()
    bst_B.load_model(xgb_B_json)

    # ★ CPU 전용 설정 강제 (예측 시점)
    #   - 일부 파라미터는 학습 전용이지만, device/nthread는 예측에 반영됨
    bst_A.set_param({"tree_method": "hist", "device": "cpu", "nthread": 3})
    bst_B.set_param({"tree_method": "hist", "device": "cpu", "nthread": 3})




    # ---- 테스트 데이터 로드 ----
    t0 = time.time()
    print("Load test data A,B ...")
    meta = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    trainA = pd.read_csv(testA_dir)
    trainB = pd.read_csv(testB_dir)
    print(f" meta={len(meta)}, Araw={len(trainA)}, Braw={len(trainB)}")
    print(f"[TIME] load: {time.time()-t0:.3f}s")

    print("Loading A.csv, B.csv ...")    


    t1 = time.time()
    A1, A2, A3, A4, A5, A6_7, A8_9_raw = run_parallel_features_A(trainA, n_jobs=n_jobs)
    B1, B2, B3, B4, B5, B6, B7, B8, B9_10_raw = run_parallel_features_B(trainB, n_jobs=n_jobs)
    print(f"[TIME] features: {time.time()-t1:.3f}s")

    A8_9_feat = A8_9_raw.copy()
    B9_10_feat = B9_10_raw.copy()

    t2 = time.time()
    A_processed = concat_A(trainA, A1, A2, A3, A4, A5, A6_7, A8_9_feat)
    B_processed = concat_B(trainB, B1, B2, B3, B4, B5, B6, B7, B8, B9_10_feat)
    print(f"[TIME] save: {time.time()-t2:.3f}s")
    print("[Preprocessing DONE]")

    A_feat = A_processed.drop(columns=["Test_id", "Test"]).copy()
    lgbm_A_features = list(lgbm_A.feature_name())
    X_lgbm_A = A_feat.reindex(columns=lgbm_A_features)

    B_feat = B_processed.drop(columns=["Test_id", "Test"]).copy()
    lgbm_B_features = list(lgbm_B.feature_name())
    X_lgbm_B = B_feat.reindex(columns=lgbm_B_features)
    # --- 예측 ---
    print("Inference (LightGBM Booster.predict)...")
    predA = lgbm_A.predict(X_lgbm_A)  # 확률값(양성 클래스) 반환
    predB = lgbm_B.predict(X_lgbm_B)  # 확률값(양성 클래스) 반환



    # --- meta와 매칭하여 out_A 생성 ---
    # meta는 Test_id, Test 두 컬럼을 갖고 있고, A만 선택
    metaA = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].copy()
    metaB = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].copy()

    # A_processed 기준으로 만든 예측과 Test_id 매칭
    subA = pd.DataFrame({
        "Test_id": A_processed["Test_id"].values,
        "prob": predA.astype(float)
    })
    subB = pd.DataFrame({
        "Test_id": B_processed["Test_id"].values,
        "prob": predB.astype(float)
    })
    probs = pd.concat([subA, subB], axis=0, ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    out_lgb = sample.merge(probs, on="Test_id", how="left")
    out_lgb["Label"] = out_lgb["prob"].astype(float).fillna(0.0)
    out_lgb = out_lgb.drop(columns=["prob"])


    # XGBoost는 DMatrix 필요. 가능하면 피처 이름 일치 권장
    dA = xgb.DMatrix(A_feat, feature_names=list(A_feat.columns))
    dB = xgb.DMatrix(B_feat, feature_names=list(B_feat.columns))

    print("Inference (XGBoost Booster.predict)...")
    predA_xgb = bst_A.predict(dA)
    predB_xgb = bst_B.predict(dB)

    subA_xgb = pd.DataFrame({"Test_id": A_processed["Test_id"].values, "prob": predA_xgb.astype(float)})
    subB_xgb = pd.DataFrame({"Test_id": B_processed["Test_id"].values, "prob": predB_xgb.astype(float)})
    probs_xgb = pd.concat([subA_xgb, subB_xgb], axis=0, ignore_index=True)

    out_xgb = sample.merge(probs_xgb, on="Test_id", how="left")
    out_xgb["Label"] = out_xgb["prob"].astype(float).fillna(0.0)
    out_xgb = out_xgb.drop(columns=["prob"])


    out = out_xgb.copy()

    # LGBM과 XGB 평균
    out["Label"] = 0.5 * out_lgb["Label"].fillna(0.0) + 0.5 * out_xgb["Label"].fillna(0.0)



    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved: {OUT_PATH} (rows={len(out)})")    

if __name__ == "__main__":
    main()