from sdv.datasets.local import load_csvs
import sys, os, time, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from functions.Validation import auc_brier_ece
from itertools import product

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
PARAM_DIR = os.path.join(parent_path, "params")
META_DIR = os.path.join(parent_path, "metadata")
SYN_DIR = os.path.join(DATA_DIR, "syn_data")
train_path = os.path.join(DATA_DIR, "train")
trainA = os.path.join(train_path, "A.csv")
processed_dir = os.path.join(DATA_DIR, "A_processed")
Atrain_labels = os.path.join(DATA_DIR, "train.csv")
trainA_pos_meta_dir = os.path.join(META_DIR, "trainA_positive_metadata.json")


datasets = load_csvs(
    folder_name=f'{processed_dir}\\',
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf-8-sig'
    })

trainA_processed = datasets['trainA_processed_fast']
Atrain_labels = pd.read_csv(Atrain_labels)

A_labels = Atrain_labels.query("Test == 'A'").copy()

trainA = pd.merge(
    A_labels, trainA_processed,
    on='Test_id', how='inner',
    validate='one_to_one', suffixes=('', '_proc')
)


# 두 Test가 동일한지 확인 후 하나만 남기기
assert (trainA['Test'] == trainA['Test_proc']).all()
trainA = trainA.drop(columns=['Test_proc'])


# --- 1) 불필요한 칼럼 제거 & X, y 분리 ---
drop_cols = ['Test_id', 'Test']  # 모델에 불필요

# print(trainA.columns)

# print(trainA['Label'].unique())

# # --- 2) Label 값 0/1 개수 세기 ---
# label_counts = trainA['Label'].value_counts()
# print("\n[Label 분포]")
# print(label_counts)




trainA = trainA.drop(columns=drop_cols)
# print(trainA.columns)




syn_pos_list = [10000, 50000, 100000, 200000, 400000]
ctgan_syn_A_pos_dict = {}
for item in syn_pos_list:
    synthe_path = os.path.join(SYN_DIR, f"ctgan_syn_A_pos_{item}.csv")
    ctgan_syn_A_pos_dict[item] = pd.read_csv(synthe_path)

# 접근
ctgan_syn_pos_10000 = ctgan_syn_A_pos_dict[10000]
ctgan_syn_pos_50000 = ctgan_syn_A_pos_dict[50000]
ctgan_syn_pos_100000 = ctgan_syn_A_pos_dict[100000]
ctgan_syn_pos_200000 = ctgan_syn_A_pos_dict[200000]
ctgan_syn_pos_400000 = ctgan_syn_A_pos_dict[400000]


####### CV #########################################################################################################



# ---------------------------
# 0) 베이스 데이터 준비
# ---------------------------
# trainA: real 전체 데이터 (Label 포함)
assert 'Label' in trainA.columns

# 합성 pos 하나 골라서 사용 (여기서는 1만짜리 예시)
syn_pos = ctgan_syn_pos_200000.copy()
syn_pos['Label'] = 1  # 혹시 안 붙어있다면 확실히 해두기

# real 전체에서 X, y 분리
X_real = trainA.drop(columns=['Label'])
y_real = trainA['Label']

print("X_real shape:", X_real.shape)
print("y_real value counts:\n", y_real.value_counts())

# 합성 pos에서도 X, y 분리
X_syn = syn_pos.drop(columns=['Label'])
y_syn = syn_pos['Label']  # 전부 1이어야 함

print("X_syn shape:", X_syn.shape)
print("y_syn unique:", y_syn.unique())






X_train = pd.concat(
    [X_real, X_syn],
    axis=0,
    ignore_index=True
)
y_train = pd.concat(
    [y_real, y_syn],
    axis=0,
    ignore_index=True
)



# ========== 0) 출력 경로 준비 ==========
OUT_DIR = os.path.join(curr_path, "output")  # 이미 위에서 정의되어 있음
os.makedirs(OUT_DIR, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")

# ========== 1) LightGBM Dataset ==========
lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

# 하이퍼파라미터 (예시값: leaf, min_data, n_jobs가 없다면 기본값 세팅)

# n_jobs = globals().get("n_jobs", 3)  # vCPU=3 환경이면 3 권장
n_jobs = 8

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 48,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
    "lambda_l1": 0.1,
    "lambda_l2": 0,    
    "num_threads": n_jobs,   # ★ lgb.train은 num_threads 사용
}

# ========== 2) 학습 ==========
# 전체 데이터를 훈련에 다 쓰고 싶다면 validation/early_stopping은 쓰지 않는 게 안전.
# (train 셋을 valid로 쓰면 의미가 없고, early_stopping도 사실상 동작 이점이 없음)
model = lgb.train(
    params=params,
    train_set=lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train],            # 로그 출력을 위해 train을 넣어둠
    valid_names=["train"],
    callbacks=[lgb.log_evaluation(20)] # early_stopping은 넣지 않음
)

print("Best iteration (if any):", model.best_iteration)  # early_stopping 없으면 None일 수 있음

# ========== 3) 저장 ==========
# 3-1) LightGBM native 모델 파일(.txt)
model_txt_path = os.path.join(OUT_DIR, f"lgbm_A.txt")
model.save_model(model_txt_path)
print("Saved:", model_txt_path)

# 3-2) 파이썬 피클(.pkl) — joblib로 직렬화 (원하면 둘 다 저장)
model_pkl_path = os.path.join(OUT_DIR, f"lgbm_A.pkl")
joblib.dump(model, model_pkl_path)
print("Saved:", model_pkl_path)