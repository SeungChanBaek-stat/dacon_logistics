# run_a_prep.py
import os, sys, time
import argparse
import tracemalloc
import numpy as np
import pandas as pd

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

    pp = PreprocessA(trainA)

    A1_feat = run_a1(pp); mem.snap("after A1")
    A2_feat = run_a2(pp); mem.snap("after A2")
    A3_feat = run_a3(pp); mem.snap("after A3")
    A4_feat = run_a4(pp); mem.snap("after A4")
    A5_feat = run_a5(pp); mem.snap("after A5")
    A6_7_feat = run_a6_7(pp); mem.snap("after A6_7")
    A8_9_feat = run_a8_9(pp); mem.snap("after A8_9")

    concat_and_save(trainA, A1_feat, A2_feat, A3_feat, A4_feat, A5_feat, A6_7_feat, A8_9_feat,
                    DATA_DIR, args.out_file)
    mem.stop()

if __name__ == "__main__":
    main()