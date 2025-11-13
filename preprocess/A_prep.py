# run_a_prep_prod.py
import os, sys, time
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler

# 프로젝트 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from functions.Preprocess import PreprocessA, parse_age_to_midpoint


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

def _run_block(df_block: pd.DataFrame):
    """블록별: A1~A7 계산 + A8_9 raw만 반환(스케일은 통합 후 1회)."""
    pp = PreprocessA(df_block)
    A1 = pp.A1_features_fast(mode='mu_sigma')
    A2 = pp.A2_features_fast(mode='mu_sigma')
    A3 = pp.A3_features_fast(mode='mu_sigma')
    A4 = pp.A4_features_fast(mode='mu_sigma')
    A5 = pp.A5_features_fast()
    A6_7 = pp.A6_7_features_fast()
    # A8_9 원본만
    cols = [c for c in ['A8-1','A8-2','A9-1','A9-2','A9-3','A9-4','A9-5'] if c in pp.df.columns]
    A8_9_raw = pp.df[cols].apply(pd.to_numeric, errors='coerce') if cols else pd.DataFrame(index=pp.df.index)
    return [A1, A2, A3, A4, A5, A6_7, A8_9_raw]

def run_parallel_features(trainA: pd.DataFrame, n_jobs: int):
    """A1~A7 병렬 + A8_9(raw), 이후 병합."""
    if n_jobs <= 1:
        pp = PreprocessA(trainA)
        A1 = pp.A1_features_fast(mode='mu_sigma')
        A2 = pp.A2_features_fast(mode='mu_sigma')
        A3 = pp.A3_features_fast(mode='mu_sigma')
        A4 = pp.A4_features_fast(mode='mu_sigma')
        A5 = pp.A5_features_fast()
        A6_7 = pp.A6_7_features_fast()
        cols = [c for c in ['A8-1','A8-2','A9-1','A9-2','A9-3','A9-4','A9-5'] if c in pp.df.columns]
        A8_9_raw = pp.df[cols].apply(pd.to_numeric, errors='coerce') if cols else pd.DataFrame(index=pp.df.index)
        return A1, A2, A3, A4, A5, A6_7, A8_9_raw

    blocks = split_by_index(trainA, n_jobs)
    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
            delayed(_run_block)(b) for b in blocks
        )

    # feature 종류별로 모으기
    A1_list, A2_list, A3_list, A4_list, A5_list, A6_7_list, A8_9raw_list = list(zip(*results))
    def _cat(lst):
        return pd.concat(lst, axis=0, ignore_index=False).sort_index()

    reindex_to = trainA.index
    A1     = _cat(A1_list).reindex(reindex_to)
    A2     = _cat(A2_list).reindex(reindex_to)
    A3     = _cat(A3_list).reindex(reindex_to)
    A4     = _cat(A4_list).reindex(reindex_to)
    A5     = _cat(A5_list).reindex(reindex_to)
    A6_7   = _cat(A6_7_list).reindex(reindex_to)
    A8_9_r = _cat(A8_9raw_list).reindex(reindex_to)
    return A1, A2, A3, A4, A5, A6_7, A8_9_r


# ---------- 저장/병합 ----------
def concat_and_save(trainA: pd.DataFrame,
                    A1_feat, A2_feat, A3_feat, A4_feat, A5_feat, A6_7_feat, A8_9_feat,
                    data_dir: str, out_name: str, fmt: str = "csv") -> str:

    # A1~A5 접두사 리네이밍
    feat_list  = [A1_feat, A2_feat, A3_feat, A4_feat, A5_feat]
    feat_names = ['A1','A2','A3','A4','A5']
    for df, name in zip(feat_list, feat_names):
        df.columns = [f"{name}-{j+1}" for j in range(df.shape[1])]

    # A6_7 컬럼명 고정
    A6_7_feat.columns = ['A6-1', 'A7-1']

    # Age
    A_age = pd.DataFrame({
        'A_age': trainA['Age'].apply(parse_age_to_midpoint).astype('Int64')
    })

    # 기본 식별 컬럼
    base_cols = trainA[['Test_id', 'Test']].reset_index(drop=True)


    dfs = [
        base_cols,
        A_age.reset_index(drop=True),
        A1_feat.reset_index(drop=True),
        A2_feat.reset_index(drop=True),
        A3_feat.reset_index(drop=True),
        A4_feat.reset_index(drop=True),
        A5_feat.reset_index(drop=True),
        A6_7_feat.reset_index(drop=True),
        A8_9_feat.reset_index(drop=True),
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
    parser.add_argument("--out-file", default="trainA_processed_fast.csv", help="저장 파일명")
    parser.add_argument("--jobs", type=int, default=1, help="병렬 작업 프로세스 수(loky). 1이면 직렬.")
    parser.add_argument("--format", choices=["csv","parquet"], default="csv", help="출력 포맷")
    args = parser.parse_args()

    curr_path  = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    DATA_DIR = args.data_dir or os.path.join(parent_path, "data")
    train_path = os.path.join(DATA_DIR, "train")
    trainA_csv = os.path.join(train_path, "A.csv")

    print("Loading A.csv ...")
    t0 = time.time()
    trainA = pd.read_csv(trainA_csv)
    print(f"[TIME] load: {time.time()-t0:.3f}s")

    print(f"Featurizing (n_jobs={args.jobs}) ...")
    t1 = time.time()
    A1, A2, A3, A4, A5, A6_7, A8_9_raw = run_parallel_features(trainA, n_jobs=args.jobs)
    print(f"[TIME] features: {time.time()-t1:.3f}s")

    # A8_9 스케일링(옵션)
    A8_9_feat = A8_9_raw.copy()
    # if args.no_scale or A8_9_raw.empty:
    #     A8_9_feat = A8_9_raw.copy()
    # else:
    #     cols = A8_9_raw.columns.tolist()
    #     scaler = StandardScaler()
    #     A8_9_feat = pd.DataFrame(scaler.fit_transform(A8_9_raw[cols]),
    #                              columns=cols, index=A8_9_raw.index)

    # 저장
    t2 = time.time()
    concat_and_save(trainA, A1, A2, A3, A4, A5, A6_7, A8_9_feat,
                    DATA_DIR, args.out_file, fmt=args.format)
    print(f"[TIME] save: {time.time()-t2:.3f}s")
    print("[DONE]")

if __name__ == "__main__":
    main()