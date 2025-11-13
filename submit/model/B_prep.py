# run_a_prep_prod.py
import os, sys, time
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler

# 프로젝트 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from functions.Preprocess import PreprocessB, parse_age_to_midpoint


# ---------- 병렬 헬퍼 ----------
def split_by_index(df: pd.DataFrame, n_blocks: int):
    """np.array_split 대체: 인덱스 슬라이스로 얕은 복사 최소화."""
    n = len(df)
    if n_blocks <= 1 or n <= 1:
        return [df]
    sizes = [n // n_blocks + (1 if i < n % n_blocks else 0) for i in range(n_blocks)]
    blocks, start = [], 0
    for sz in sizes:
        end = start + sz
        if sz > 0:
            blocks.append(df.iloc[start:end])
        start = end
    return blocks

def _run_block_B(df_block: pd.DataFrame):
    """블록별: B1~B8 계산 + B9_10 raw만 반환(스케일은 통합 후 1회)."""
    pp = PreprocessB(df_block)
    B1 = pp.B1_features_fast(mode='mu_sigma')
    B2 = pp.B2_features_fast(mode='mu_sigma')
    B3 = pp.B3_features_fast(mode='mu_sigma')
    B4 = pp.B4_features_fast(mode='mu_sigma')
    B5 = pp.B5_features_fast(mode='mu_sigma')
    B6 = pp.B6_features_fast(mode='mu_sigma')
    B7 = pp.B7_features_fast(mode='mu_sigma')
    B8 = pp.B8_features_fast(mode='mu_sigma')
    # A8_9 원본만
    cols = [c for c in ['B9-1','B9-2','B9-3','B9-4','B9-5','B10-1','B10-2','B10-3','B10-4','B10-5','B10-6'] if c in pp.df.columns]
    B9_10_raw = pp.df[cols].apply(pd.to_numeric, errors='coerce') if cols else pd.DataFrame(index=pp.df.index)
    return [B1, B2, B3, B4, B5, B6, B7, B8, B9_10_raw]

def run_parallel_features_B(trainB: pd.DataFrame, n_jobs: int):
    """B1~B8 계산 + B9_10 raw, 이후 병합."""
    if n_jobs <= 1:
        pp = PreprocessB(trainB)
        B1 = pp.B1_features_fast(mode='mu_sigma')
        B2 = pp.B2_features_fast(mode='mu_sigma')
        B3 = pp.B3_features_fast(mode='mu_sigma')
        B4 = pp.B4_features_fast(mode='mu_sigma')
        B5 = pp.B5_features_fast(mode='mu_sigma')
        B6 = pp.B6_features_fast(mode='mu_sigma')
        B7 = pp.B7_features_fast(mode='mu_sigma')
        B8 = pp.B8_features_fast(mode='mu_sigma')
        # A8_9 원본만
        cols = [c for c in ['B9-1','B9-2','B9-3','B9-4','B9-5','B10-1','B10-2','B10-3','B10-4','B10-5','B10-6'] if c in pp.df.columns]
        B9_10_raw = pp.df[cols].apply(pd.to_numeric, errors='coerce') if cols else pd.DataFrame(index=pp.df.index)
        return B1, B2, B3, B4, B5, B6, B7, B8, B9_10_raw

    blocks = split_by_index(trainB, n_jobs)
    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
            delayed(_run_block_B)(b) for b in blocks
        )

    # feature 종류별로 모으기
    B1_list, B2_list, B3_list, B4_list, B5_list, B6_list, B7_list, B8_list, B9_10raw_list = list(zip(*results))
    def _cat(lst):
        return pd.concat(lst, axis=0, ignore_index=False).sort_index()

    reindex_to = trainB.index
    B1     = _cat(B1_list).reindex(reindex_to)
    B2     = _cat(B2_list).reindex(reindex_to)
    B3     = _cat(B3_list).reindex(reindex_to)
    B4     = _cat(B4_list).reindex(reindex_to)
    B5     = _cat(B5_list).reindex(reindex_to)
    B6     = _cat(B6_list).reindex(reindex_to)
    B7     = _cat(B7_list).reindex(reindex_to)
    B8     = _cat(B8_list).reindex(reindex_to)
    B9_10_r = _cat(B9_10raw_list).reindex(reindex_to)
    return B1, B2, B3, B4, B5, B6, B7, B8, B9_10_r


# ---------- 저장/병합 ----------
def concat_and_save_B(trainB: pd.DataFrame,
                    B1_feat, B2_feat, B3_feat, B4_feat, B5_feat, B6_feat, B7_feat, B8_feat, B9_10_feat,
                    data_dir: str, out_name: str, fmt: str = "csv") -> str:

    # A1~A5 접두사 리네이밍
    feat_list  = [B1_feat, B2_feat, B3_feat, B4_feat, B5_feat, B6_feat, B7_feat, B8_feat]
    feat_names = ['B1','B2','B3','B4','B5','B6','B7','B8']
    for df, name in zip(feat_list, feat_names):
        df.columns = [f"{name}-{j+1}" for j in range(df.shape[1])]

    # A6_7 컬럼명 고정
    B9_10_feat.columns = ['B9-1','B9-2','B9-3','B9-4','B9-5','B10-1','B10-2','B10-3','B10-4','B10-5','B10-6']

    # Age
    B_age = pd.DataFrame({
        'B_age': trainB['Age'].apply(parse_age_to_midpoint).astype('Int64')
    })

    # 기본 식별 컬럼
    base_cols = trainB[['Test_id', 'Test']].reset_index(drop=True)

    dfs = [
        base_cols,
        B_age.reset_index(drop=True),
        B1_feat.reset_index(drop=True),
        B2_feat.reset_index(drop=True),
        B3_feat.reset_index(drop=True),
        B4_feat.reset_index(drop=True),
        B5_feat.reset_index(drop=True),
        B6_feat.reset_index(drop=True),
        B7_feat.reset_index(drop=True),
        B8_feat.reset_index(drop=True),
        B9_10_feat.reset_index(drop=True),
    ]
    final_df = pd.concat(dfs, axis=1)

    out_path = os.path.join(data_dir, out_name)
    if fmt.lower() == "parquet":
        # 빠르고 용량 효율적
        final_df.to_parquet(out_path if out_path.endswith(".parquet") else out_path + ".parquet",
                            index=False, engine="pyarrow")
    else:
        final_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"[OK] Saved: {out_path}")
    return out_path


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="data 폴더 절대/상대 경로 (기본: ../data)")
    parser.add_argument("--out-file", default="trainB_processed_fast.csv", help="저장 파일명")
    parser.add_argument("--jobs", type=int, default=1, help="병렬 작업 프로세스 수(loky). 1이면 직렬.")
    parser.add_argument("--format", choices=["csv","parquet"], default="csv", help="출력 포맷")
    args = parser.parse_args()

    curr_path  = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    DATA_DIR = args.data_dir or os.path.join(parent_path, "data")
    train_path = os.path.join(DATA_DIR, "train")
    trainB_csv = os.path.join(train_path, "B.csv")

    print("Loading B.csv ...")
    t0 = time.time()
    trainB = pd.read_csv(trainB_csv)
    print(f"[TIME] load: {time.time()-t0:.3f}s")

    print(f"Featurizing (n_jobs={args.jobs}) ...")
    t1 = time.time()
    B1, B2, B3, B4, B5, B6, B7, B8, B9_10_raw = run_parallel_features_B(trainB, n_jobs=args.jobs)
    print(f"[TIME] features: {time.time()-t1:.3f}s")

    B9_10_feat = B9_10_raw.copy()

    # 저장
    t2 = time.time()
    concat_and_save_B(trainB, B1, B2, B3, B4, B5, B6, B7, B8, B9_10_feat,
                    DATA_DIR, args.out_file, fmt=args.format)
    print(f"[TIME] save: {time.time()-t2:.3f}s")
    print("[DONE]")

if __name__ == "__main__":
    main()