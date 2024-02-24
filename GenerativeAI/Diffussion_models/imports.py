import torch
import numpy as np
import random
import torchvision

SEED = 42
random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)
