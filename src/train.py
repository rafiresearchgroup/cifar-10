
from evaluate import evaluate
import torch

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    # loop over batches

    #num_of_batches = epochs / batch_size
    #for batch_idx in range(num_of_batches):
    #    images = train_loader()

    total_loss = 0
    correct = 0
    n = len(train_loader.dataset)

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()           # zero the gradients from prev batch
        pred = model(images)            # forward pass ← get predictions
        loss = criterion(pred, labels)  # compute loss: mean over 128 images (batch size)
        loss.backward()                 # backward pass (walks the graph backwards and computes d(loss)/d(param) for every parameter using the chain rule)
        optimizer.step()                # uses .grad to update weights "param = param - lr * param.grad"

        total_loss += loss.item()* images.size(0) #  treats last batch  (64-image) same as 128-image batch
        correct += (pred.argmax(1) ==labels).sum().item()

    return total_loss/n, correct/n




def train(model, train_loader, test_loader, optimizer, criterion, device, cfg):
    #loop over epochs

    best_acc = 0
    for epoch_idx in range(cfg.epochs):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # evaluate on test --> could be a function: evalute(model, test_loader,criterion,device)
        test_loss, test_acc = evaluate(model, test_loader,criterion,device)
        # test_images = test_loader.images
        # test_labels = test_loader.labels
        # test_pred = model(test_images) 
        # test_loss = criterion(test_pred, test_labels)
        # test_acc = (test_pred.argmax(1) == test_pred).sum()

        print(f"Epoch: {epoch_idx+1:02d}/{cfg.epochs} || " f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% || " f" Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%")
        # :02d    # ✅ integer: 2 digits wide, zero padded
        #:.1f    # ✅ float: just decimal places, no width before the dot
        # :08.2f   # 8 chars wide, zero padded  (no truncation)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), cfg.save_dir + cfg.save_name)
            print(f"  → saved best model at epoch {epoch_idx+1}: {best_acc*100:.1f}%")           



if __name__ == "__main__":

    print("test the training")
    import os
    import torch

    print(os.getcwd())
    import torch.nn as nn
    import torch.optim as optim
    from model import CIFAR10Net, CIFAR10NetDeeper

    from dataset import get_data_loader
    data_dir = "/Users/rdallatorre/repos/cifar-10/data/"

    train_loader, test_loader = get_data_loader(data_dir)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CIFAR10Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(model, train_loader, test_loader, optimizer, criterion, device, epochs=10)
