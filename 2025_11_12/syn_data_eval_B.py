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
trainB = os.path.join(train_path, "B.csv")
processed_dir = os.path.join(DATA_DIR, "B_processed")
Btrain_labels = os.path.join(DATA_DIR, "train.csv")
ctgan_param_dir = os.path.join(PARAM_DIR, "ctgan_synthesizer_B.pkl")
trainB_pos_meta_dir = os.path.join(META_DIR, "trainB_positive_metadata.json")



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

# --- 3) Label == 1 인 데이터만 추출 ---
trainB_pos = trainB[trainB['Label'] == 1].copy()
print(f"\ntrainB_neg shape: {trainB_pos.shape}")


trainB_pos_ = trainB_pos.drop(columns=drop_cols)
print(trainB.head())
print(trainB.columns)












metadata = Metadata.load_from_json(filepath=trainB_pos_meta_dir)


syn_pos_list = [10000, 50000, 100000]
# syn_neg_list = 

for item in syn_pos_list:
    t0 = time.time()
    print(f"Analyzing ctgan_syn_B_pos_{item} data...")
    synthe_path = os.path.join(SYN_DIR, f"ctgan_syn_B_pos_{item}.csv")
    pseudo_data = pd.read_csv(synthe_path)
    print("pseudo_data dim : ", pseudo_data.shape)
    diagnostic_report = run_diagnostic(
        real_data=trainB_pos_,
        synthetic_data=pseudo_data,
        metadata=metadata)
    quality_report = evaluate_quality(trainB_pos_, pseudo_data, metadata)
    details = quality_report.get_details('Column Shapes')
    print(details.sort_values('Score').head(10))  # 가장 점수 낮은 컬럼 10개    
    

    print(f"Analyzing ctgan_syn_B_pos_{item} data complete : {time.time()-t0:.3f}s")




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
# fig = get_column_plot(
#     real_data=trainB_pos_,
#     synthetic_data=pseudo_data,
#     metadata=metadata,
#     column_name="B3-10"
# )
    
# fig.show()

# # 3. plot the data
# fig = get_column_plot(
#     real_data=trainB_pos_,
#     synthetic_data=pseudo_data,
#     metadata=metadata,
#     column_name="B3-3"
# )
    
# fig.show()


# # 3. plot the data
# fig = get_column_plot(
#     real_data=trainB_pos_,
#     synthetic_data=pseudo_data,
#     metadata=metadata,
#     column_name="B8-5"
# )
    
# fig.show()