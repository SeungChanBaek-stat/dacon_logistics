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

Araw = pd.read_csv(train_A)
Araw_cols = Araw.columns
print(Araw_cols)



# print(Araw.head(5))

A1_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
Araw_A1 = None
for item in A1_cols:
    Araw_A1 = pd.concat([Araw_A1, Araw[item]], axis = 1)

print(Araw_A1.head(5))
print(Araw_A1.shape)