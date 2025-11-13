from sdv.utils import load_synthesizer
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
import sys, os, time
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot

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

# --- 3) Label == 1 인 데이터만 추출 ---
trainA_pos = trainA[trainA['Label'] == 1].copy()
print(f"\ntrainA_neg shape: {trainA_pos.shape}")


trainA_pos_ = trainA_pos.drop(columns=drop_cols)
print(trainA.head())
print(trainA.columns)












metadata = Metadata.load_from_json(filepath=trainA_pos_meta_dir)


syn_pos_list = [100000, 200000, 400000]
# syn_neg_list = 

for item in syn_pos_list:
    t0 = time.time()
    print(f"Analyzing ctgan_syn_A_pos_{item} data...")
    synthe_path = os.path.join(SYN_DIR, f"ctgan_syn_A_pos_{item}.csv")
    pseudo_data = pd.read_csv(synthe_path)
    diagnostic_report = run_diagnostic(
        real_data=trainA_pos_,
        synthetic_data=pseudo_data,
        metadata=metadata)
    quality_report = evaluate_quality(trainA_pos_, pseudo_data, metadata)
    details = quality_report.get_details('Column Shapes')
    print(details.sort_values('Score').head(10))  # 가장 점수 낮은 컬럼 10개    
    

    print(f"Analyzing ctgan_syn_A_pos_{item} data complete : {time.time()-t0:.3f}s")




# synthe_path = os.path.join(OUT_DIR, "synthetic_data_10000.csv")
# print("data synthesizing...")

# print("data synthesizing complete")

# pseudo_data = pd.read_csv(synthe_path)


# diagnostic_report = run_diagnostic(
#     real_data=trainA_pos_,
#     synthetic_data=pseudo_data,
#     metadata=metadata)


# # 2. measure the statistical similarity
# quality_report = evaluate_quality(trainA_pos_, pseudo_data, metadata)

# details = quality_report.get_details('Column Shapes')
# print(details.sort_values('Score').head(10))  # 가장 점수 낮은 컬럼 10개


# 3. plot the data
fig = get_column_plot(
    real_data=trainA_pos_,
    synthetic_data=pseudo_data,
    metadata=metadata,
    column_name="A5-11"
)
    
fig.show()

# 3. plot the data
fig = get_column_plot(
    real_data=trainA_pos_,
    synthetic_data=pseudo_data,
    metadata=metadata,
    column_name="A5-8"
)
    
fig.show()


# 3. plot the data
fig = get_column_plot(
    real_data=trainA_pos_,
    synthetic_data=pseudo_data,
    metadata=metadata,
    column_name="A1-3"
)
    
fig.show()