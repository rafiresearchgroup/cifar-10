# import torch
# import torch.nn as nn
# from src.config import Config
# from src.model import CIFAR10NetDeeper
# from src.evaluate import evaluate
# from src.dataset import get_data_loader

# def test_evaluate_output_range():
#     # create random model
#     # get test loader
#     # call evaluate()
#     # assert loss > 0
#     # assert 0 <= acc <= 1

# def test_random_model_accuracy():
#     # random model should get ~10% accuracy
#     # assert acc < 0.2  (above 0% but below 20%)

# def test_evaluate_returns_tuple():
#     # evaluate should return (loss, acc)
#     # assert isinstance(result, tuple)
#     # assert len(result) == 2



import torch
import torch.nn as nn
from src.config import Config
from src.model import CIFAR10NetDeeper
from src.evaluate import evaluate
from src.dataset import get_data_loader

device = 'cpu'  # always use cpu for tests — consistent across machines

def test_evaluate_returns_tuple():
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    _, test_loader = get_data_loader(cfg)
    criterion = nn.CrossEntropyLoss()

    result = evaluate(model, test_loader, criterion, device)

    assert isinstance(result, tuple)
    assert len(result) == 2

def test_evaluate_output_range():
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    _, test_loader = get_data_loader(cfg)
    criterion = nn.CrossEntropyLoss()

    loss, acc = evaluate(model, test_loader, criterion, device)

    assert loss > 0
    assert 0.0 <= acc <= 1.0

def test_random_model_accuracy():
    # untrained model should be near random (10% for 10 classes)
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    _, test_loader = get_data_loader(cfg)
    criterion = nn.CrossEntropyLoss()

    _, acc = evaluate(model, test_loader, criterion, device)

    assert acc < 0.2   # above 0% but well below trained performance

def test_random_model_loss():
    # untrained model loss should be near log(10) = 2.3
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    _, test_loader = get_data_loader(cfg)
    criterion = nn.CrossEntropyLoss()

    loss, _ = evaluate(model, test_loader, criterion, device)

    assert 1.5 < loss < 4.0  # loose bounds around log(10)=2.3

def test_model_in_eval_mode():
    # evaluate() should set model to eval mode
    cfg = Config()
    model = CIFAR10NetDeeper(cfg).to(device)
    _, test_loader = get_data_loader(cfg)
    criterion = nn.CrossEntropyLoss()

    model.train()  # force train mode first
    evaluate(model, test_loader, criterion, device)

    assert not model.training  # should be False after evaluate()
