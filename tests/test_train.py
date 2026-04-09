import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.model import CIFAR10NetDeeper
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.dataset import get_data_loader

device = 'cpu'

def get_small_loader(cfg, n=500):
    """small subset for fast tests"""
    train_loader, _ = get_data_loader(cfg)
    subset = Subset(train_loader.dataset, range(n))
    return DataLoader(subset, batch_size=cfg.batch_size)


def test_loss_decreases_after_one_epoch():
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    loader = get_small_loader(cfg)

    loss_before, _ = evaluate(model, loader, criterion, device)
    train_one_epoch(model, loader, optimizer, criterion, device)
    loss_after, _ = evaluate(model, loader, criterion, device)

    assert loss_after < loss_before


def test_accuracy_above_random_after_one_epoch():
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    loader = get_small_loader(cfg)

    train_one_epoch(model, loader, optimizer, criterion, device)
    _, acc = evaluate(model, loader, criterion, device)

    assert acc > 0.15  # above random (10%)


def test_weights_change_after_one_batch():
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    loader = get_small_loader(cfg)

    # snapshot weights before
    weights_before = model.features[0].weight.clone()

    train_one_epoch(model, loader, optimizer, criterion, device)

    # weights should have changed
    weights_after = model.features[0].weight
    assert not torch.equal(weights_before, weights_after)


def test_loss_below_random_after_one_epoch():
    # after one epoch loss should be below log(10)=2.3 (random baseline)
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    loader = get_small_loader(cfg)

    train_one_epoch(model, loader, optimizer, criterion, device)
    loss, _ = evaluate(model, loader, criterion, device)

    assert loss < 2.3  # below random baseline


def test_train_one_epoch_returns_metrics():
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    loader = get_small_loader(cfg)

    result = train_one_epoch(model, loader, optimizer, criterion, device)

    assert isinstance(result, tuple)
    assert len(result) == 2

    loss, acc = result
    assert loss > 0
    assert 0.0 <= acc <= 1.0