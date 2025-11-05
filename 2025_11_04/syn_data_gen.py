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
ctgan_param_dir = os.path.join(PARAM_DIR, "ctgan_synthesizer.pkl")
trainA_pos_meta_dir = os.path.join(META_DIR, "trainA_positive_metadata.json")


metadata = Metadata.load_from_json(filepath=trainA_pos_meta_dir)

synthesizer_load_path = os.path.join(ctgan_param_dir)
print("load synthesizer")
synthesizer = load_synthesizer(filepath=synthesizer_load_path)
print("loading synthesizer complete")

syn_pos_list = [200000, 400000]
# syn_neg_list = 

for item in syn_pos_list:
    synthe_path = os.path.join(SYN_DIR, f"ctgan_syn_pos_{item}.csv")
    print(f"ctgan_syn_pos_{item} data synthesizing...")
    t0 = time.time()
    synthetic_data = synthesizer.sample(num_rows=item)
    synthetic_data.to_csv(synthe_path, index=False)
    print(f"ctgan_syn_pos_{item} data synthesizing complete : {time.time()-t0:.3f}s")
    synthesizer.reset_sampling()




