from sdv.datasets.local import load_csvs
import sys, os, time
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
trainB = os.path.join(train_path, "B.csv")
processed_dir = os.path.join(DATA_DIR, "B_processed")
Btrain_labels = os.path.join(DATA_DIR, "train.csv")
ctgan_param_dir = os.path.join(PARAM_DIR, "ctgan_B_synthesizer.pkl")
trainB_pos_meta_dir = os.path.join(META_DIR, "trainB_positive_metadata.json")


datasets = load_csvs(
    folder_name=f'{processed_dir}\\',
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf-8-sig'
    })

trainB_processed = datasets['trainB_processed_fast']
Btrain_labels = pd.read_csv(Btrain_labels)

B_labels = Btrain_labels.query("Test == 'B'").copy()

trainB = pd.merge(
    B_labels, trainB_processed,
    on='Test_id', how='inner',
    validate='one_to_one', suffixes=('', '_proc')
)


# 두 Test가 동일한지 확인 후 하나만 남기기
assert (trainB['Test'] == trainB['Test_proc']).all()
trainB = trainB.drop(columns=['Test_proc'])


# --- 1) 불필요한 칼럼 제거 & X, y 분리 ---
drop_cols = ['Test_id', 'Test']  # 모델에 불필요

# print(trainA.columns)

# print(trainA['Label'].unique())

# # --- 2) Label 값 0/1 개수 세기 ---
# label_counts = trainA['Label'].value_counts()
# print("\n[Label 분포]")
# print(label_counts)




trainB = trainB.drop(columns=drop_cols)
# print(trainA.columns)




syn_pos_list = [10000, 50000, 100000]
ctgan_syn_B_pos_dict = {}
for item in syn_pos_list:
    synthe_path = os.path.join(SYN_DIR, f"ctgan_syn_B_pos_{item}.csv")
    ctgan_syn_B_pos_dict[item] = pd.read_csv(synthe_path)

# 접근
ctgan_syn_pos_10000 = ctgan_syn_B_pos_dict[10000]
ctgan_syn_pos_50000 = ctgan_syn_B_pos_dict[50000]
ctgan_syn_pos_100000 = ctgan_syn_B_pos_dict[100000]




####### CV #########################################################################################################



# ---------------------------
# 0) 베이스 데이터 준비
# ---------------------------
# trainA: real 전체 데이터 (Label 포함)
assert 'Label' in trainB.columns

# 합성 pos 하나 골라서 사용 (여기서는 1만짜리 예시)
syn_pos = ctgan_syn_pos_100000.copy()
syn_pos['Label'] = 1  # 혹시 안 붙어있다면 확실히 해두기

# real 전체에서 X, y 분리
X_real = trainB.drop(columns=['Label'])
y_real = trainB['Label']

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
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 2) fold별로 train/valid 분리 + 합성 pos 붙이기
# ---------------------------
fold_indices = []  # 나중에 분석에 쓰고 싶으면 저장

# cpu 사용갯수
n_jobs = 8

# best num_leaves = 48, min_data_in_leaf = 100
num_leaves_list = [32, 48, 64]
min_data_in_leaf_list = [100, 150, 200, 250]

# best learning_rate = 0.02, num_boost_round = 1000
# lr_list = [0.02, 0.03, 0.04, 0.05]


# num_boost_round_list = [200, 300, 400, 500, 600, 700, 800]
num_boost_round_list = [500, 1000, 1500]

# best λ₁=0.1, λ₂=0
# lambda_l1_list = [0.0, 1e-3, 1e-2, 1e-1]
# lambda_l2_list = [0.0, 1e-3, 1e-2, 1e-1]
results = []

# for num_leaves, min_data_in_leaf in product(num_leaves_list, min_data_in_leaf_list):
# for num_leaves, min_data_in_leaf in product(num_leaves_list, min_data_in_leaf_list):
for boost_round in num_boost_round_list:

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
            "learning_rate": 0.02,
            "num_leaves": 48,
            "min_data_in_leaf": 100, # ← 여기 추가
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
            "verbosity": -1,
            "lambda_l1": 0.01,
            "lambda_l2": 0,            
            "n_jobs": n_jobs
        }


        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=boost_round,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            # **lgb.callback = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]**
            # , lgb.early_stopping(stopping_rounds=100)
            callbacks = [lgb.log_evaluation(100), lgb.early_stopping(stopping_rounds=300)]
            # callbacks = [lgb.log_evaluation(100)]
            # early_stopping_rounds=???  # 쓸지 말지는 우리가 결정
        )

        # 4) 검증셋 확률 예측
        y_pred_proba = model.predict(X_valid_real, num_iteration=model.best_iteration)  # 0~1 확률
        
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

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    # results.append((num_leaves, min_data_in_leaf, mean_score, std_score))
    # print(f"[num_leaves={num_leaves}, min_data_in_leaf={min_data_in_leaf}] mean score: {mean_score:.6f}, std score: {std_score:.6f}")
    results.append((boost_round, mean_score, std_score))
    print(f"[boost_round={boost_round}] mean score: {mean_score:.6f}, std score: {std_score:.6f}")


# === 모든 실험 끝난 후 ===
# results_df = pd.DataFrame(results, columns=["num_leaves", "min_data_in_leaf", "mean_score", "std_score"])
results_df = pd.DataFrame(results, columns=["boost_round", "mean_score", "std_score"])

# mean_score 기준 오름차순 정렬 (즉, 낮을수록 좋은 점수)
results_df = results_df.sort_values("mean_score", ascending=True).reset_index(drop=True)

print("\n===== CV 결과 요약 =====")
print(results_df.to_string(index=False, float_format="%.6f"))

# best hyperparameter 출력
best_row = results_df.iloc[0]
# print(f"\n✅ Best params → boost_round={int(best_row.num_leaves)}, min_data_in_leaf={int(best_row.min_data_in_leaf)} "
print(f"\n✅ Best params → boost_round={int(best_row.boost_round)} "
      f"| mean_score={best_row.mean_score:.6f} | std={best_row.std_score:.6f}")

# # 결과 CSV로 저장 (선택사항)
# OUT_DIR = os.path.join(curr_path, "output")
# os.makedirs(OUT_DIR, exist_ok=True)
# csv_path = os.path.join(OUT_DIR, "lgb_cv_results3.csv")
# results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
# print(f"\n결과 저장 완료: {csv_path}")