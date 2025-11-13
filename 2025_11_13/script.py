# run_a_prep_prod.py
import os, sys, time
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler

# 프로젝트 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from model.Preprocess import PreprocessA, PreprocessB, parse_age_to_midpoint, split_by_index
from model.Preprocess import _run_block_A, run_parallel_features_A, concat_A, _run_block_B, run_parallel_features_B, concat_B





# ---------- main ----------
def main():
    n_jobs = 3

    # TEST_DIR = "./data"
    curr_path  = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    DATA_DIR = os.path.join(parent_path, "data")
    train_path = os.path.join(DATA_DIR, "train")
    trainA_csv = os.path.join(train_path, "A.csv")

    TEST_DIR  = "./data"              # test.csv, A.csv, B.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"             # lgbm_A.pkl, lgbm_B.pkl 위치
    OUT_DIR   = "./output"
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH  = os.path.join(OUT_DIR, "submission.csv")



    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    meta = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    Araw = pd.read_csv(os.path.join(TEST_DIR, "./test/A.csv"))
    Braw = pd.read_csv(os.path.join(TEST_DIR, "./test/B.csv"))
    print(f" meta={len(meta)}, Araw={len(Araw)}, Braw={len(Braw)}")


    print("Loading A.csv, B.csv ...")
    t0 = time.time()
    trainA = pd.read_csv(Araw)
    trainB = pd.read_csv(Braw)
    print(f"[TIME] load: {time.time()-t0:.3f}s")


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

if __name__ == "__main__":
    main()