import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from typing import List, Optional
from functions.Preprocess import PreprocessA, parse_age_to_midpoint
import time

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
train_path = os.path.join(DATA_DIR, "train")
trainA = os.path.join(train_path, "A.csv")

start = time.time()
trainA = pd.read_csv(trainA)
end = time.time()
print(f"trainA loading : {end - start:.5f} sec")



pp = PreprocessA(trainA)

start = time.time()
A1_split = pp.A1_parse_and_split()
A1_feat = pp.A1_features(A1_split, mode = 'mu_sigma')
end = time.time()
print(f"A1_feat {end - start:.5f} sec")

start = time.time()
A2_split = pp.A2_parse_and_split()
A2_feat = pp.A2_features(A2_split, mode = 'mu_sigma')
end = time.time()
print(f"A2_feat {end - start:.5f} sec")

start = time.time()
A3_split = pp.A3_parse_and_split()
A3_feat = pp.A3_features(A3_split, mode = 'mu_sigma')
end = time.time()
print(f"A3_feat {end - start:.5f} sec")

start = time.time()
A4_split = pp.A4_parse_and_split()
A4_feat = pp.A4_features(A4_split, mode = 'mu_sigma')
end = time.time()
print(f"A4_feat {end - start:.5f} sec")

start = time.time()
A5_split = pp.A5_parse_and_split()
A5_feat = pp.A5_features(A5_split, mode = 'mu_sigma')
end = time.time()
print(f"A5_feat {end - start:.5f} sec")

start = time.time()
A6_7_split = pp.A6_7_parse_and_split()
A6_7_feat = pp.A6_7_features(A6_7_split, mode = 'mu_sigma')
end = time.time()
print(f"A6_7_feat {end - start:.5f} sec")

A8_9_feat = pp.A8_9_features()

# feature DataFrame들을 리스트로 묶기
feat_list = [A1_feat, A2_feat, A3_feat, A4_feat, A5_feat]
feat_names = ['A1', 'A2', 'A3', 'A4', 'A5']

# A1~A5 컬럼명 일괄 변경
for i, (df, name) in enumerate(zip(feat_list, feat_names), start=1):
    new_cols = [f'{name}-{j+1}' for j in range(df.shape[1])]
    df.columns = new_cols

# A6_7_feat은 수동 처리
A6_7_feat.columns = ['A6-1', 'A7-1']



# --- A_age DataFrame 생성 ---
A_age = pd.DataFrame({
    'A_age': trainA['Age'].apply(parse_age_to_midpoint).astype('Int64')  # 결측 허용 정수 타입
})

# --- concat 대상 베이스 컬럼 ---
base_cols = trainA[['Test_id', 'Test']].reset_index(drop=True)

# --- 피처 DF 인덱스 정렬(안전장치) ---
dfs = [
    base_cols,
    A_age.reset_index(drop=True),
    A1_feat.reset_index(drop=True),
    A2_feat.reset_index(drop=True),
    A3_feat.reset_index(drop=True),
    A4_feat.reset_index(drop=True),
    A5_feat.reset_index(drop=True),
    A6_7_feat.reset_index(drop=True),
    A8_9_feat.reset_index(drop=True),  # 건드리지 않음
]

# --- 최종 병합 ---
final_df = pd.concat(dfs, axis=1)

# --- 저장 ---
# os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(DATA_DIR, "trainA_processed.csv")
final_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"[OK] Saved: {out_path}")
print(final_df.head())