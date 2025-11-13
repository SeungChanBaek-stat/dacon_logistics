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
trainA = os.path.join(train_path, "A.csv")
processed_dir = os.path.join(DATA_DIR, "A_processed")
Atrain_labels = os.path.join(DATA_DIR, "train.csv")


# print(trainA)
# assume that my_folder contains a CSV file named 'guests.csv'
datasets = load_csvs(
    folder_name=f'{processed_dir}\\',
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf-8-sig'
    })

trainA_processed = datasets['trainA_processed_fast']
Atrain_labels = pd.read_csv(Atrain_labels)
# print(trainA_processed)
# print(type(trainA_processed))

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
drop_cols = ['Test_id', 'Test', 'Label']  # 모델에 불필요

print(trainA.columns)

print(trainA['Label'].unique())

# --- 2) Label 값 0/1 개수 세기 ---
label_counts = trainA['Label'].value_counts()
print("\n[Label 분포]")
print(label_counts)

# --- 3) Label == 0 인 데이터만 추출 ---
trainA_pos = trainA[trainA['Label'] == 1].copy()
print(f"\ntrainA_neg shape: {trainA_pos.shape}")


trainA_pos_ = trainA_pos.drop(columns=drop_cols)
print(trainA.head())
print(trainA.columns)


metadata = Metadata.detect_from_dataframe(
    data=trainA_pos_,
    table_name='trainA_positive')
metadata.save_to_json(os.path.join(OUT_DIR, "trainA_positive_metadata.json"))

synthesizer = CTGANSynthesizer(
    metadata, # required
    epochs=500,
    verbose=True,
    cuda=True
)

synthesizer.fit(trainA_pos_)
synthesizer.get_loss_values()

fig = synthesizer.get_loss_values_plot()
fig.show()


save_path = os.path.join(OUT_DIR, "ctgan_synthesizer_A.pkl")
synthesizer.save(filepath=save_path)
print(f"Synthesizer 모델이 저장되었습니다: {save_path}")
# print(metadata)


# # the data is available under the file name
# data = datasets['guests']