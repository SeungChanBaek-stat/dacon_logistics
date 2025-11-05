from sdv.datasets.local import load_csvs
import sys, os, time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from functions.Validation import auc_brier_ece

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
ctgan_param_dir = os.path.join(PARAM_DIR, "ctgan_synthesizer.pkl")
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
ctgan_syn_pos_dict = {}
for item in syn_pos_list:
    synthe_path = os.path.join(SYN_DIR, f"ctgan_syn_pos_{item}.csv")
    ctgan_syn_pos_dict[item] = pd.read_csv(synthe_path)

# 접근
ctgan_syn_pos_10000 = ctgan_syn_pos_dict[10000]
ctgan_syn_pos_50000 = ctgan_syn_pos_dict[50000]
ctgan_syn_pos_100000 = ctgan_syn_pos_dict[100000]
ctgan_syn_pos_200000 = ctgan_syn_pos_dict[200000]
ctgan_syn_pos_400000 = ctgan_syn_pos_dict[400000]











# 1) real 데이터에서 pos / neg 분리
trainA_pos_real = trainA[trainA['Label'] == 1].copy()
trainA_neg_real = trainA[trainA['Label'] == 0].copy()

print("[real pos shape]:", trainA_pos_real.shape)
print("[real neg shape]:", trainA_neg_real.shape)

# 2) 합성 pos 데이터들에 Label=1 컬럼 추가
ctgan_syn_pos_10000 = ctgan_syn_pos_10000.copy()
ctgan_syn_pos_50000 = ctgan_syn_pos_50000.copy()
ctgan_syn_pos_100000 = ctgan_syn_pos_100000.copy()
ctgan_syn_pos_200000 = ctgan_syn_pos_200000.copy()
ctgan_syn_pos_400000 = ctgan_syn_pos_400000.copy()

for df in [ctgan_syn_pos_10000, ctgan_syn_pos_50000, ctgan_syn_pos_100000, ctgan_syn_pos_200000, ctgan_syn_pos_400000]:
    df['Label'] = 1  # 모두 양성 클래스

print("[syn_pos_10000 shape]:", ctgan_syn_pos_10000.shape)
print("[syn_pos_50000 shape]:", ctgan_syn_pos_50000.shape)
print("[syn_pos_100000 shape]:", ctgan_syn_pos_100000.shape)
print("[syn_pos_200000 shape]:", ctgan_syn_pos_200000.shape)
print("[syn_pos_400000 shape]:", ctgan_syn_pos_400000.shape)

# # (선택) 한 번에 쓸 합성 pos를 하나로 합치고 싶다면:
syn_pos_all = pd.concat(
    [ctgan_syn_pos_10000, ctgan_syn_pos_50000, ctgan_syn_pos_100000],
    axis=0,
    ignore_index=True
)
print("[syn_pos_all shape]:", syn_pos_all.shape)
















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

# ---------------------------
# 1) Stratified 10-fold 설정
# ---------------------------
skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 2) fold별로 train/valid 분리 + 합성 pos 붙이기
# ---------------------------
fold_indices = []  # 나중에 분석에 쓰고 싶으면 저장

fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_real, y_real), start=1):
    print(f"\n===== Fold {fold} =====")
    t0 = time.time()
    
    # real 데이터 기준 train/valid 분리
    X_train_real = X_real.iloc[train_idx].copy()
    y_train_real = y_real.iloc[train_idx].copy()
    X_valid_real = X_real.iloc[valid_idx].copy()
    y_valid_real = y_real.iloc[valid_idx].copy()
    
    # print("[real train] shape:", X_train_real.shape, "pos=", (y_train_real == 1).sum(), "neg=", (y_train_real == 0).sum())
    # print("[real valid] shape:", X_valid_real.shape, "pos=", (y_valid_real == 1).sum(), "neg=", (y_valid_real == 0).sum())
    
    # --- 합성 pos를 train에만 붙이기 ---
    X_train_fold = pd.concat(
        [X_train_real, X_syn],
        axis=0,
        ignore_index=True
    )
    y_train_fold = pd.concat(
        [y_train_real, y_syn],
        axis=0,
        ignore_index=True
    )
    # # --- 합성 pos 쓰지 않는 경우 ---
    # X_train_fold, y_train_fold = X_train_real, y_train_real
    # print("[train + synthetic] shape:", X_train_fold.shape, 
    #       "pos=", (y_train_fold == 1).sum(), 
    #       "neg=", (y_train_fold == 0).sum())
    
    # 원하면 나중에 다시 쓰려고 index 저장
    fold_indices.append({
        "train_idx": train_idx,
        "valid_idx": valid_idx
    })
    
    # -----------------------------------

    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    lgb_valid = lgb.Dataset(X_valid_real, label=y_valid_real, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",         # 모니터링용
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbosity": -1,
    }

    lgb.early_stopping

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        valid_names=['valid'],
        # **lgb.callback = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]**
        callbacks = [lgb.log_evaluation(500)]
        # early_stopping_rounds=???  # 쓸지 말지는 우리가 결정
    )

    # 4) 검증셋 확률 예측
    y_pred_proba = model.predict(X_valid_real)  # 0~1 확률
    
    # 5) 대회 메트릭 계산
    answer_df = pd.DataFrame({
        "id": np.arange(len(y_valid_real)),
        "Label": y_valid_real.values.astype(int)
    })
    
    submission_df = pd.DataFrame({
        "id": np.arange(len(y_pred_proba)),
        "Label": y_pred_proba.astype(float)
    })
    
    fold_score = auc_brier_ece(answer_df, submission_df)
    print(f"Fold {fold} combined score: {fold_score:.6f}, elapsed : {time.time()-t0:.3f}s")
    fold_scores.append(fold_score)

print("\nCV mean combined score:", np.mean(fold_scores))
print("CV std combined score :", np.std(fold_scores))