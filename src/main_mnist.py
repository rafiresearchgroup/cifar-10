
import torch.nn as nn
import torch.optim as optim
import torch

from model import CIFAR10NetShallow, CIFAR10NetDeeper
from dataset import get_data_loader, get_data_loader_mnist
from train import train
from config import Config
from evaluate import evaluate

cfg = Config()
#data_dir = "/Users/rdallatorre/repos/cifar-10/data/"

# read data
train_loader, test_loader = get_data_loader(cfg)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")  # should print 'mps'

# create model
#model = CIFAR10NetShallow(cfg).to(device)  
model = CIFAR10NetDeeper(cfg).to(device)  

# train (on CIFAR)
TRAIN_CIFAR = False
EVAL_MNIST = 1
FINETUNE_MNIST = 1
MODEL_INIT_RANDOM = 1
criterion = nn.CrossEntropyLoss()

if TRAIN_CIFAR:
    optimizer = optim.SGD(model.parameters(), cfg.lr, cfg.momentum)
    train(model, train_loader, test_loader, optimizer, criterion, device, cfg)
    print("trained model on CIFAR")

else:
    model.load_state_dict(torch.load("trained_models/best_model_cifar.pth"))
    print("load trained model on CIFAR")
    #Evaluate on CIFAR

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"CIFAR: Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")



if MODEL_INIT_RANDOM:
    model = CIFAR10NetDeeper(cfg).to(device)
    print("IGNORE CIFAR training")
    #Evaluate on CIFAR

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"CIFAR: Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")



# Load MNIST
cfg_mnist = Config()
cfg_mnist.save_name = "best_model_mnist.pth"  # ← override

train_loader_mnist, test_loader_mnist = get_data_loader_mnist(Config())

if FINETUNE_MNIST:
    # after loading CIFAR weights
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # 10x smaller lr

    # use your existing train() function but with mnist loaders
    train(model, train_loader_mnist, test_loader_mnist, 
        optimizer, criterion, device, cfg_mnist)
    print("fine-tuned model on MNIST")
    

if EVAL_MNIST:

    model.eval()

    #Evaluate on CIFAR

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"CIFAR: Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")

    #Evaluate on MNIST
    # evaluate on MNIST
    test_loss, test_acc = evaluate(model, test_loader_mnist, criterion, device)
    print(f"MNIST: Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")







    # total_loss= 0
    # correct = 0
    # n = len(test_loader_mnist.dataset)
    # with torch.no_grad():
    #     for images, labels in test_loader_mnist:

    #         images = images.to(device) # torch.Size([128, 1, 28, 28]) vs. torch.Size([128, 3, 32, 32])
    #         labels = labels.to(device) # torch.Size([128])
    #         # NxCxWxH feature map

    #         #images = 1 - images          # ← invert here, before anything else

    #         images_ = images
    #         # images_ = images.repeat(1, 3, 1, 1)  
    #         # zp = nn.ZeroPad2d((2,2,2,2))  #<-- do the padding in the transform, before normalization
    #         # images__ = zp(images_)

    #         # debug
    #         # print(f"Input shape: {images_.shape}")      # should be (128,3,32,32)
    #         # print(f"Input mean:  {images_.mean():.4f}") # should be near 0
    #         # print(f"Input std:   {images_.std():.4f}")  # should be near 1
    #         # print(f"Input min:   {images_.min():.4f}")  # check for extreme values
    #         # print(f"Input max:   {images_.max():.4f}")

    #         # pred = model(images__)            # forward pass ← get predictions
    #         pred = model(images_)            # forward pass ← get predictions
    #         loss = criterion(pred, labels)  # compute loss, mean over batch of 128

    #         acc = (pred.argmax(1) ==labels).sum().item()
    #         print(f"Loss = {loss.item():.4f} | Correct = {acc:02d} #")
    #         total_loss += loss.item()*  images_.size(0)
    #         correct += acc

    #     print(f"Loss =  {total_loss/n:.4f} ? Accuracy {correct/n*100:.1f} %" )




