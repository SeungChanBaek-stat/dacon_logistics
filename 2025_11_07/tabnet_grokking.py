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
import random
from pytorch_tabnet.callbacks import Callback

class PeriodicCheckpoint(Callback):
    """
    매 period(100) 에포크마다:
    1) 해당 period 구간 내 best 모델 1개 저장
    2) period 마지막 에포크(100, 200, 300...) 모델 1개 저장
    
    구간 내에서는 메모리에만 best 정보를 보관하고, 디스크 저장은 period 배수마다만 실행
    """
    def __init__(self, save_dir, period=100, metric_name="valid_auc"):
        self.save_dir = save_dir
        self.period = period
        self.metric_name = metric_name
        os.makedirs(save_dir, exist_ok=True)
        
        # 현재 블록 내 best 정보를 메모리에만 보관
        self.current_block = 1
        self.best_metric_in_block = -float("inf")
        self.best_model_in_block = None  # 모델 state dict를 임시 저장
        self.best_epoch_in_block = None
        
        print(f"[PeriodicCheckpoint] Initialized")
        print(f"  - Checkpoint period: {period} epochs")
        print(f"  - Metric: {metric_name}")
        print(f"  - Save dir: {save_dir}\n")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_metric = logs.get(self.metric_name, None)
        
        if current_metric is None:
            return
        
        actual_epoch = epoch + 1  # TabNet은 0부터 시작
        current_block = ((epoch) // self.period) + 1
        
        # 새로운 블록 시작 시 초기화
        if current_block != self.current_block:
            print(f"\n{'='*70}")
            print(f"[Block {current_block}] Starting epochs {(current_block-1)*self.period + 1} - {current_block*self.period}")
            print(f"{'='*70}")
            self.current_block = current_block
            self.best_metric_in_block = -float("inf")
            self.best_model_in_block = None
            self.best_epoch_in_block = None
        
        # 블록 내 best 업데이트 (메모리에만 저장)
        if current_metric > self.best_metric_in_block:
            self.best_metric_in_block = current_metric
            self.best_epoch_in_block = actual_epoch
            # 모델의 state_dict 복사본을 메모리에 저장
            self.best_model_in_block = {
                'epoch': actual_epoch,
                'metric': current_metric,
                'network_state': self.trainer.network.state_dict().copy()
            }
            print(f"  [Block {self.current_block}] New best at epoch {actual_epoch}: {self.metric_name}={current_metric:.6f}")
        
        # period 배수 에포크에 도달하면 디스크에 저장
        if actual_epoch % self.period == 0:
            print(f"\n{'*'*70}")
            print(f"[SAVING] Reached epoch {actual_epoch} - Saving 2 models...")
            print(f"{'*'*70}")
            
            # 1) 현재(period 배수) 에포크 모델 저장
            checkpoint_path = os.path.join(
                self.save_dir,
                f"epoch{actual_epoch:04d}_checkpoint"
            )
            self.trainer.save_model(checkpoint_path)
            print(f"  ✓ Saved checkpoint: epoch {actual_epoch}")
            
            # 2) 블록 내 best 모델 저장
            if self.best_model_in_block is not None:
                best_path = os.path.join(
                    self.save_dir,
                    f"epoch{actual_epoch:04d}_block_best_epoch{self.best_epoch_in_block:04d}_auc{self.best_metric_in_block:.4f}"
                )
                
                # best 모델의 state_dict를 현재 모델에 로드한 후 저장
                current_state = self.trainer.network.state_dict().copy()  # 현재 상태 백업
                self.trainer.network.load_state_dict(self.best_model_in_block['network_state'])
                self.trainer.save_model(best_path)
                self.trainer.network.load_state_dict(current_state)  # 원래 상태로 복원
                
                print(f"  ✓ Saved block best: epoch {self.best_epoch_in_block} (auc={self.best_metric_in_block:.6f})")
            
            print(f"{'*'*70}\n")
            
            # 블록 요약
            print(f"[Block {self.current_block} Summary]")
            print(f"  - Epochs: {(self.current_block-1)*self.period + 1} - {actual_epoch}")
            print(f"  - Best epoch: {self.best_epoch_in_block}")
            print(f"  - Best {self.metric_name}: {self.best_metric_in_block:.6f}\n")
    
    def on_train_end(self, logs=None):
        """학습 종료 시 최종 요약"""
        print(f"\n{'='*70}")
        print(f"[Training Complete]")
        if self.best_epoch_in_block is not None:
            print(f"Last block ({self.current_block}) best: epoch {self.best_epoch_in_block}, auc={self.best_metric_in_block:.6f}")
        print(f"All checkpoints saved to: {self.save_dir}")
        print(f"{'='*70}\n")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)






# 1. 학습 곡선 시각화 개선 (그로킹 현상 관찰용)
def plot_grokking_metrics(history, save_path=None):
    """그로킹 현상을 관찰하기 위한 상세 시각화"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1) Train/Valid Loss (로그 스케일)
    ax = axes[0, 0]
    ax.plot(history['loss'], label='Train Loss', alpha=0.7)
    if 'valid_loss' in history:
        ax.plot(history['valid_loss'], label='Valid Loss', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2) Train/Valid AUC (그로킹 관찰용)
    ax = axes[0, 1]
    ax.plot(history['train_auc'], label='Train AUC', alpha=0.7)
    ax.plot(history['valid_auc'], label='Valid AUC', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC over Time (Grokking Check)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 그로킹 타이밍 표시
    if len(history['valid_auc']) > 10:
        valid_auc = np.array(history['valid_auc'])
        # Valid AUC가 처음으로 0.6을 넘는 시점
        grok_threshold = 0.6
        grok_epochs = np.where(valid_auc > grok_threshold)[0]
        if len(grok_epochs) > 0:
            first_grok = grok_epochs[0]
            ax.axvline(x=first_grok, color='g', linestyle='--', alpha=0.5, 
                      label=f'Grok at epoch {first_grok}')
            ax.legend()
    
    # 3) Generalization Gap (overfitting 체크)
    ax = axes[1, 0]
    gap = np.array(history['train_auc']) - np.array(history['valid_auc'])
    ax.plot(gap, color='purple', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train AUC - Valid AUC')
    ax.set_title('Generalization Gap (Overfitting Check)')
    ax.grid(True, alpha=0.3)
    
    # 4) Learning Rate
    ax = axes[1, 1]
    ax.plot(history['lr'], color='orange', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved plot] {save_path}")
    
    plt.show()


# 2. 그로킹 감지 콜백
class GrokkingDetector(Callback):
    """그로킹 현상을 감지하고 기록하는 콜백"""
    
    def __init__(self, threshold_improvement=0.05, window_size=50):
        """
        Args:
            threshold_improvement: Valid AUC의 급격한 상승을 감지하는 임계값
            window_size: 이동 평균 윈도우 크기
        """
        self.threshold = threshold_improvement
        self.window_size = window_size
        self.valid_auc_history = []
        self.grokking_detected = False
        self.grokking_epoch = None
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        valid_auc = logs.get('valid_auc', None)
        
        if valid_auc is not None:
            self.valid_auc_history.append(valid_auc)
            
            # 충분한 히스토리가 쌓이면 그로킹 체크
            if len(self.valid_auc_history) >= self.window_size and not self.grokking_detected:
                recent_avg = np.mean(self.valid_auc_history[-self.window_size:])
                old_avg = np.mean(self.valid_auc_history[-2*self.window_size:-self.window_size]) \
                         if len(self.valid_auc_history) >= 2*self.window_size else 0.5
                
                improvement = recent_avg - old_avg
                
                # 급격한 성능 향상 감지
                if improvement > self.threshold:
                    self.grokking_detected = True
                    self.grokking_epoch = epoch + 1
                    print(f"\n{'='*60}")
                    print(f"🎯 GROKKING DETECTED at Epoch {self.grokking_epoch}!")
                    print(f"Valid AUC improved by {improvement:.4f} over last {self.window_size} epochs")
                    print(f"{'='*60}\n")


# 3. 개선된 TabNet 파라미터 (그로킹 관찰용)
def get_grokking_tabnet_params():
    """그로킹 실험에 적합한 TabNet 파라미터"""
    import torch
    
    return {
        # 모델 용량 (그로킹은 충분한 모델 용량이 필요)
        "n_d": 64,           # 48 -> 64로 증가
        "n_a": 64,           # 48 -> 64로 증가
        "n_steps": 5,
        "gamma": 1.5,        # 1.3 -> 1.5 (좀 더 aggressive attention)
        "n_independent": 2,
        "n_shared": 2,
        
        # 정규화 (그로킹은 강한 정규화가 도움됨)
        "lambda_sparse": 1e-3,
        
        # 옵티마이저 (낮은 learning rate + weight decay)
        "optimizer_fn": torch.optim.AdamW,
        "optimizer_params": dict(
            lr=5e-4,           # 1e-3 -> 5e-4 (더 느리게)
            weight_decay=0.05  # 0.02 -> 0.05 (더 강한 정규화)
        ),
        
        # 스케줄러 (patience를 길게)
        "scheduler_params": {
            "factor": 0.5,     # 0.05 -> 0.5 (덜 aggressive)
            "patience": 100,   # patience 추가
            "min_lr": 1e-6
        },
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        
        "mask_type": "entmax",
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }


# 4. 조기 종료 비활성화 함수
def train_for_grokking(clf, X_train, y_train, X_valid, y_valid, 
                       max_epochs=2000, checkpoint_period=100,
                       checkpoint_dir="./checkpoints"):
    """
    그로킹 관찰을 위한 학습 (조기 종료 없음)
    """
    from pytorch_tabnet.callbacks import Callback
    
    # 콜백 설정
    periodic_cb = PeriodicCheckpoint(
        save_dir=checkpoint_dir,
        period=checkpoint_period,
        metric_name="valid_auc"
    )
    
    grokking_cb = GrokkingDetector(
        threshold_improvement=0.05,
        window_size=50
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Grokking Experiment")
    print(f"Max Epochs: {max_epochs}")
    print(f"Checkpoint Period: {checkpoint_period}")
    print(f"Early Stopping: DISABLED (for grokking observation)")
    print(f"{'='*60}\n")
    
    clf.fit(
        X_train=X_train, 
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs,
        patience=max_epochs,  # 사실상 조기 종료 비활성화
        batch_size=4096,
        virtual_batch_size=256,
        num_workers=0,
        weights=1,
        drop_last=False,
        augmentations=None,
        callbacks=[periodic_cb, grokking_cb],
    )
    
    return clf, grokking_cb










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


    tabnet_params = get_grokking_tabnet_params()
    clf = TabNetClassifier(**tabnet_params)


    num_epochs = 3000  # 그로킹 맛보기는 300부터 시작하여 나중에 늘려도 됨
    # patience = 50

    # 원하면 TabNet 내부 augmentations도 쓸 수 있음
    # aug = ClassificationSMOTE(p=0.2)
    aug = None  # 일단 끔

    checkpoint_dir = os.path.join(OUT_DIR, "tabnet_checkpoints")


    # 2) 학습
    clf, grokking_cb = train_for_grokking(
        clf, X_train, y_train, X_valid, y_valid,
        max_epochs=num_epochs,
        checkpoint_period=100,
        checkpoint_dir=checkpoint_dir
    )

    # 3) 시각화
    plot_path = os.path.join(OUT_DIR, "grokking_analysis.png")
    plot_grokking_metrics(clf.history, save_path=plot_path)
    
    # 4) 그로킹 정보 출력
    if grokking_cb.grokking_detected:
        print(f"\n✅ Grokking occurred at epoch {grokking_cb.grokking_epoch}")
    else:
        print(f"\n❌ No clear grokking detected within {num_epochs} epochs")    

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
    set_seed(42)
    main()