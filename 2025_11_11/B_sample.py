import numpy as np
import pandas as pd
import os

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
MODEL_DIR = os.path.join(curr_path, "model")
OUT_DIR   = os.path.join(curr_path, "output")
train_path = os.path.join(DATA_DIR, "train")
train_A = os.path.join(train_path, "A.csv")
train_B = os.path.join(train_path, "B.csv")

# print(curr_path)
# print(parent_path)
# print(train_path)

Braw = pd.read_csv(train_B)
Braw_cols = Braw.columns
print(Braw_cols)

Bexercise = Braw[0:2000]

# 저장 경로 설정
sample_path = os.path.join(DATA_DIR, "B_sample.csv")

# UTF-8-sig 인코딩으로 저장 (엑셀에서 한글 깨짐 방지)
Bexercise.to_csv(sample_path, index=False, encoding="utf-8-sig")

print(f"샘플 데이터가 저장되었습니다: {sample_path}")




# # print(Araw.head(5))

# A1_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
# A1 = None
# for item in A1_cols:
#     A1 = pd.concat([A1, Araw[item]], axis = 1)

# print(A1.head(5))
# print(A1.shape)

# n = len(A1)

# A1_list
# for i in range(n):
#     A1['A1-1', i]