import numpy as np
import lightgbm as lgb
print(np.__version__)
print(lgb.__version__)

import os
curr_path = os.getcwd()
print(curr_path)

parent_path = os.path.dirname(curr_path)
print(parent_path)