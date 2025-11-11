from pytorch_tabnet.tab_model import TabNetClassifier
from sdv.datasets.local import load_csvs
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import numpy as np
np.random.seed(42)
import scipy
import sys, os, time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from functions.Validation import auc_brier_ece
from itertools import product
import matplotlib.pyplot as plt

def main():

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














    # 1. real 데이터 기준으로 통일된 통계 사용
    real_median = X_real.median()

    # 2. real / syn / valid 모두 같은 기준으로 채움
    X_real = X_real.fillna(real_median)
    X_syn = X_syn.fillna(real_median)

    # 3. 이후 train/valid split 재실행
    X_train_real, X_valid_real, y_train_real, y_valid_real = train_test_split(
        X_real, y_real, test_size=0.2, stratify=y_real, random_state=42
    )

    # 4. 합성 pos 붙이기
    X_train_all = pd.concat([X_train_real, X_syn], ignore_index=True)
    y_train_all = pd.concat([y_train_real, y_syn], ignore_index=True)









    # print("== NaN check: real / syn / combined ==")
    # print("X_real NaN 개수:", X_real.isna().sum().sum())
    # print("X_syn NaN 개수:", X_syn.isna().sum().sum())

    # X_train_all = pd.concat([X_train_real, X_syn], axis=0, ignore_index=True)
    # print("X_train_all NaN 개수:", X_train_all.isna().sum().sum())


    # nan_cols = X_train_all.columns[X_train_all.isna().any()]
    # print("NaN 있는 컬럼:", nan_cols.tolist())
    # print(X_train_all[nan_cols].isna().sum())








    print("[real train] shape:", X_train_real.shape,
        "pos=", (y_train_real == 1).sum(),
        "neg=", (y_train_real == 0).sum())
    print("[real valid] shape:", X_valid_real.shape,
        "pos=", (y_valid_real == 1).sum(),
        "neg=", (y_valid_real == 0).sum())

    # ---------------------------
    # 1) train에만 합성 pos 붙이기
    # ---------------------------

    X_train_all = pd.concat([X_train_real, X_syn], ignore_index=True)
    y_train_all = pd.concat([y_train_real, y_syn], ignore_index=True)

    print("[train + synthetic] shape:", X_train_all.shape,
        "pos=", (y_train_all == 1).sum(),
        "neg=", (y_train_all == 0).sum())

    features = list(X_real.columns)  # 전체 피처 이름

    X_train = X_train_all[features].values.astype(np.float32)
    y_train = y_train_all.values.astype(int)

    X_valid = X_valid_real[features].values.astype(np.float32)
    y_valid = y_valid_real.values.astype(int)

    print("X_train shape:", X_train.shape, "y_train pos=", (y_train == 1).sum())
    print("X_valid shape:", X_valid.shape, "y_valid pos=", (y_valid == 1).sum())


    tabnet_params = {
        # 카테고리 없으니 cat_idxs, cat_dims는 생략
        "n_d": 32,
        "n_a": 32,
        "n_steps": 5,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-4,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2),  # 예제는 2e-2, 필요하면 1e-3 등으로 조절
        "scheduler_params": {"step_size": 50, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "mask_type": "entmax",  # 예제와 동일
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }

    clf = TabNetClassifier(**tabnet_params)

    max_epochs = 300  # 그로킹 맛보기는 300부터 시작하여 나중에 늘려도 됨
    patience = 50

    # 원하면 TabNet 내부 augmentations도 쓸 수 있음
    # aug = ClassificationSMOTE(p=0.2)
    aug = None  # 일단 끔

    clf.fit(
        X_train=X_train, 
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],  # train/valid 둘 다 모니터링
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=0,
        weights=1,       # 예제에서는 1 → class weight 안 씀
        drop_last=False,
        augmentations=aug,
    )


    # plot losses
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.plot(clf.history['loss'])
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1,3,2)
    plt.plot(clf.history['train_auc'], label="train_auc")
    plt.plot(clf.history['valid_auc'], label="valid_auc")
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(clf.history['lr'])
    plt.title("Learning Rate")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.show()



    # 확률 예측
    preds_valid = clf.predict_proba(X_valid)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)
    print(f"TabNet valid AUC: {valid_auc:.6f}")

    from functions.Validation import auc_brier_ece

    answer_df = pd.DataFrame({
        "id": np.arange(len(y_valid)),
        "Label": y_valid.astype(int)
    })

    submission_df = pd.DataFrame({
        "id": np.arange(len(preds_valid)),
        "Label": preds_valid[:, 1].astype(float)
    })

    combined_score = auc_brier_ece(answer_df, submission_df)
    print(f"TabNet valid combined score: {combined_score:.6f}")


if __name__ == "__main__":
    main()