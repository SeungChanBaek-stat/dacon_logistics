from pytorch_tabnet.tab_model import TabNetClassifier
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())