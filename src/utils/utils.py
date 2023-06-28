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


def clear_cuda_cache():
    print("\n Clearing cuda cache...")
    torch.cuda.empty_cache()