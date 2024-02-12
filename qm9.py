import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from models.radar import radar_factory
import math
from models.sc_net import SCnet
import pickle
from dataset.tmQM_data import tmQM_dataset,tmQM_wrapper
import yaml
from torch_geometric.datasets import QM9

# Download

dp = QM9(root='./data/QM9/')

# config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
# dataloader = tmQM_wrapper(config['path'],config['batch_size'],separated=config['separated'],qm9=dp**config['data'])
# train,valid,test = dataloader.get_data_loaders()