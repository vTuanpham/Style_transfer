import sys
sys.path.insert(0,r'./')
import random
import torch
import numpy as np


def set_seed(value):
    print("\n Random Seed: ", value)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)