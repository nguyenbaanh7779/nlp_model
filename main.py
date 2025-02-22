import os
import sys
import config_env
sys.path.insert(2, os.path.join(config_env.ROOT_PATH, "src"))

import datasets

import utils

dataset = datasets.load_from_disk(os.path.join(config_env.ROOT_PATH, "data/raw/xsum"))
print(dataset)