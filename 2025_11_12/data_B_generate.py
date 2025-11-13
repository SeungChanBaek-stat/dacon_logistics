from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
import sys, os
import pandas as pd
from sdv.single_table import CTGANSynthesizer

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
DATA_DIR  = os.path.join(parent_path, "data")
OUT_DIR   = os.path.join(curr_path, "output")
train_path = os.path.join(DATA_DIR, "train")
processed_dir = os.path.join(DATA_DIR, "B_processed")
Btrain_labels = os.path.join(DATA_DIR, "train.csv")


# print(trainA)
# assume that my_folder contains a CSV file named 'guests.csv'
datasets = load_csvs(
    folder_name=f'{processed_dir}\\',
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf-8-sig'
    })

trainB_processed = datasets['trainB_processed_fast']
Btrain_labels = pd.read_csv(Btrain_labels)
# print(trainA_processed)
# print(type(trainA_processed))

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
drop_cols = ['Test_id', 'Test', 'Label']  # 모델에 불필요

print(trainB.columns)

print(trainB['Label'].unique())

# --- 2) Label 값 0/1 개수 세기 ---
label_counts = trainB['Label'].value_counts()
print("\n[Label 분포]")
print(label_counts)

# --- 3) Label == 0 인 데이터만 추출 ---
trainB_pos = trainB[trainB['Label'] == 1].copy()
print(f"\ntrainB_neg shape: {trainB_pos.shape}")


trainB_pos_ = trainB_pos.drop(columns=drop_cols)
print(trainB.head())
print(trainB.columns)





metadata = Metadata.detect_from_dataframe(
    data=trainB_pos_,
    table_name='trainB_positive')
metadata.save_to_json(os.path.join(OUT_DIR, "trainB_positive_metadata.json"))

synthesizer = CTGANSynthesizer(
    metadata, # required
    epochs=500,
    verbose=True,
    cuda=True
)

synthesizer.fit(trainB_pos_)
synthesizer.get_loss_values()

fig = synthesizer.get_loss_values_plot()
fig.show()


save_path = os.path.join(OUT_DIR, "ctgan_synthesizer_B.pkl")
synthesizer.save(filepath=save_path)
print(f"Synthesizer 모델이 저장되었습니다: {save_path}")