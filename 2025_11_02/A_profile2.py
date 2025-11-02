# run_a_prep.py
import os, sys, time
import argparse
import tracemalloc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import math
from sklearn.preprocessing import StandardScaler

# 프로젝트 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from functions.Preprocess import PreprocessA, parse_age_to_midpoint  # <- 이미 있으신 모듈
# 선택: from sklearn.preprocessing import StandardScaler  # A8_9 에서 내부 사용

# ---------------------------
# 유틸: 타이머/메모리 데코레이터
# ---------------------------
def timed(section_name: str):
    def _wrap(fn):
        def inner(*args, **kwargs):
            t0 = time.time()
            out = fn(*args, **kwargs)
            dt = time.time() - t0
            print(f"[TIME] {section_name}: {dt:.3f} sec")
            return out
        return inner
    return _wrap

class MemProbe:
    """tracemalloc로 섹션별 메모리 스냅샷을 찍어 peak/현재 사용량을 보여줌"""
    def __init__(self, enable: bool):
        self.enable = enable
        if self.enable and not tracemalloc.is_tracing():
            tracemalloc.start()

    def snap(self, tag: str):
        if not self.enable:
            return
        cur, peak = tracemalloc.get_traced_memory()
        # 바이트 → MB
        print(f"[MEM ] {tag}: current={cur/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")

    def stop(self):
        if self.enable and tracemalloc.is_tracing():
            tracemalloc.stop()





def split_by_index(df: pd.DataFrame, n_blocks: int):
    """np.array_split 대신 인덱스 범위로 자르기(불필요한 복사 최소화)."""
    n = len(df)
    sizes = [n // n_blocks + (1 if i < n % n_blocks else 0) for i in range(n_blocks)]
    blocks = []
    start = 0
    for sz in sizes:
        end = start + sz
        if sz > 0:
            blocks.append(df.iloc[start:end])
        start = end
    return blocks

def _run_block(df_block: pd.DataFrame):
    """각 블록에서 A1~A7 만 계산. A8_9는 'raw'만 반환하고 표준화는 나중에 한 번에."""
    pp = PreprocessA(df_block)
    A1 = pp.A1_features_fast(mode='mu_sigma')
    A2 = pp.A2_features_fast(mode='mu_sigma')
    A3 = pp.A3_features_fast(mode='mu_sigma')
    A4 = pp.A4_features_fast(mode='mu_sigma')
    A5 = pp.A5_features_fast()
    A6_7 = pp.A6_7_features_fast()
    # A8_9는 "스케일 전 원본"만 넘김 (표준화는 병합 후 한 번만)
    A8_9_raw = pp.df[['A8-1','A8-2','A9-1','A9-2','A9-3','A9-4','A9-5']].copy()
    A8_9_raw = A8_9_raw.apply(pd.to_numeric, errors='coerce')
    return [A1, A2, A3, A4, A5, A6_7, A8_9_raw]

def run_parallel_features(trainA: pd.DataFrame, n_jobs: int):
    """블록 병렬로 A1~A7 + A8_9(raw) 계산 후 병합.
       A8_9는 병합 뒤 단 한 번만 StandardScaler로 표준화."""
    if n_jobs <= 1:
        # 직렬 경로 (기존과 동일)
        pp = PreprocessA(trainA)
        A1 = pp.A1_features_fast(mode='mu_sigma')
        A2 = pp.A2_features_fast(mode='mu_sigma')
        A3 = pp.A3_features_fast(mode='mu_sigma')
        A4 = pp.A4_features_fast(mode='mu_sigma')
        A5 = pp.A5_features_fast()
        A6_7 = pp.A6_7_features_fast()
        A8_9_raw = pp.df[['A8-1','A8-2','A9-1','A9-2','A9-3','A9-4','A9-5']].apply(pd.to_numeric, errors='coerce')
        return A1, A2, A3, A4, A5, A6_7, A8_9_raw

    # 블록 나누기
    blocks = split_by_index(trainA, n_jobs)

    # 과도한 내부 스레딩 방지 (MKL/BLAS 1스레드)
    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
            delayed(_run_block)(b) for b in blocks
        )

    # results는 [ [A1_b1, A2_b1, ...], [A1_b2, ...], ... ]
    # 축을 바꿔서 feature별 리스트로 만든 뒤 concat
    feat_lists = list(zip(*results))  # 길이 7의 튜플(A1 list, A2 list, ...)
    A1_list, A2_list, A3_list, A4_list, A5_list, A6_7_list, A8_9raw_list = feat_lists

    def _cat(lst):
        # ignore_index=False로 원 인덱스 유지, 이후 원본 인덱스 순서로 정렬
        dfc = pd.concat(lst, axis=0, ignore_index=False)
        return dfc.sort_index()

    A1 = _cat(A1_list)
    A2 = _cat(A2_list)
    A3 = _cat(A3_list)
    A4 = _cat(A4_list)
    A5 = _cat(A5_list)
    A6_7 = _cat(A6_7_list)
    A8_9_raw = _cat(A8_9raw_list)

    # 혹시라도 인덱스가 섞였으면 trainA 인덱스로 재색인(누락은 NaN)
    reindex_to = trainA.index
    A1 = A1.reindex(reindex_to)
    A2 = A2.reindex(reindex_to)
    A3 = A3.reindex(reindex_to)
    A4 = A4.reindex(reindex_to)
    A5 = A5.reindex(reindex_to)
    A6_7 = A6_7.reindex(reindex_to)
    A8_9_raw = A8_9_raw.reindex(reindex_to)

    return A1, A2, A3, A4, A5, A6_7, A8_9_raw





# ---------------------------
# 파이프라인
# ---------------------------
@timed("Load A.csv")
def load_train_a(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@timed("A1 features")
def run_a1(pp: PreprocessA):
    return pp.A1_features_fast(mode='mu_sigma')

@timed("A2 features")
def run_a2(pp: PreprocessA):
    return pp.A2_features_fast(mode='mu_sigma')

@timed("A3 features")
def run_a3(pp: PreprocessA):
    return pp.A3_features_fast(mode='mu_sigma')

@timed("A4 features")
def run_a4(pp: PreprocessA):
    return pp.A4_features_fast(mode='mu_sigma')

@timed("A5 features")
def run_a5(pp: PreprocessA):
    return pp.A5_features_fast()

@timed("A6_7 features")
def run_a6_7(pp: PreprocessA):
    return pp.A6_7_features_fast()

@timed("A8_9 features")
def run_a8_9(pp: PreprocessA):
    return pp.A8_9_features()

@timed("Concat & Save")
def concat_and_save(trainA: pd.DataFrame,
                    A1_feat, A2_feat, A3_feat, A4_feat, A5_feat, A6_7_feat, A8_9_feat,
                    data_dir: str, out_name: str) -> str:

    # 컬럼명 간소화
    feat_list  = [A1_feat, A2_feat, A3_feat, A4_feat, A5_feat]
    feat_names = ['A1','A2','A3','A4','A5']
    for df, name in zip(feat_list, feat_names):
        df.columns = [f"{name}-{j+1}" for j in range(df.shape[1])]

    A6_7_feat.columns = ['A6-1', 'A7-1']  # 수동 지정

    # Age 파생
    A_age = pd.DataFrame({
        'A_age': trainA['Age'].apply(parse_age_to_midpoint).astype('Int64')
    })

    # 기본 컬럼
    base_cols = trainA[['Test_id', 'Test']].reset_index(drop=True)

    # 인덱스 정렬 및 합치기
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

    # 저장
    out_path = os.path.join(data_dir, out_name)
    final_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved: {out_path}")
    print(final_df.head())
    return out_path

# ---------------------------
# main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="data 폴더 절대/상대 경로 (기본: ../data)")
    parser.add_argument("--out-file", default="trainA_processed_fast.csv", help="저장 파일명")
    parser.add_argument("--mem", action="store_true", help="tracemalloc 메모리 프로브 활성화")
    parser.add_argument("--jobs", type=int, default=1, help="병렬 작업 프로세스 수(loky). 1이면 직렬.")
    args = parser.parse_args()

    curr_path  = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    DATA_DIR = args.data_dir or os.path.join(parent_path, "data")
    train_path = os.path.join(DATA_DIR, "train")
    trainA_csv = os.path.join(train_path, "A.csv")

    mem = MemProbe(enable=args.mem)

    print("trainA loading...")
    trainA = load_train_a(trainA_csv)
    mem.snap("after load")

    # === 병렬 피처 추출 (A1~A7 + A8_9 raw) ===
    print(f"Featureizing with n_jobs={args.jobs} ...")
    t0 = time.time()
    A1_feat, A2_feat, A3_feat, A4_feat, A5_feat, A6_7_feat, A8_9_raw = run_parallel_features(trainA, n_jobs=args.jobs)
    print(f"[TIME] Parallel featureizing: {time.time()-t0:.3f} sec")
    mem.snap("after parallel")

    # === A8_9 표준화는 합친 뒤 한 번만 ===
    scaler = StandardScaler()
    A8_9_cols = [c for c in ['A8-1','A8-2','A9-1','A9-2','A9-3','A9-4','A9-5'] if c in A8_9_raw.columns]
    A8_9_feat = pd.DataFrame(
        scaler.fit_transform(A8_9_raw[A8_9_cols]),
        columns=A8_9_cols,
        index=A8_9_raw.index
    )

    mem.snap("after A8_9 scale")

    # === 저장 및 마무리 ===
    concat_and_save(trainA, A1_feat, A2_feat, A3_feat, A4_feat, A5_feat, A6_7_feat, A8_9_feat,
                    DATA_DIR, args.out_file)
    mem.stop()

if __name__ == "__main__":
    main()