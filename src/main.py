import torch.nn as nn
import torch.optim as optim
import torch

from model import CIFAR10NetShallow, CIFAR10NetDeeper
from dataset import get_data_loader, get_data_loader_mnist
from train import train
from config import Config

cfg = Config()
#data_dir = "/Users/rdallatorre/repos/cifar-10/data/"

# read data
train_loader, test_loader = get_data_loader(cfg)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")  # should print 'mps'

# create model
# model = CIFAR10NetShallow(cfg).to(device)  
model = CIFAR10NetDeeper(cfg).to(device)  

# train

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), cfg.lr, cfg.momentum)
train(model, train_loader, test_loader, optimizer, criterion, device, cfg)