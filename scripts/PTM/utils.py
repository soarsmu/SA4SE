import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW

from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import random
import time

from sklearn.metrics import accuracy_score, classification_report

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# Datasets
from pathlib import Path
data_folder=Path('/sa4se/data/')

api_train=data_folder/'api-train.pkl'
api_test=data_folder/'api-test.pkl'

gh_train=data_folder/'gh-train.pkl'
gh_test=data_folder/'gh-test.pkl'

jira_train=data_folder/'jira-train.pkl'
jira_test=data_folder/'jira-test.pkl'

so_train=data_folder/'so-train.pkl'
so_test=data_folder/'so-test.pkl'

app_train=data_folder/'app-train.pkl'
app_test=data_folder/'app-test.pkl'

cr_train=data_folder/'cr-train.pkl'
cr_test=data_folder/'cr-test.pkl'

# Hyperparameters
MAX_LEN=256
BATCH_SIZE=16
EPOCHS=4
LEARNING_RATE=2e-5