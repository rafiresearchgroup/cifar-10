import torch

def evaluate(model, test_loader,criterion,device):
    

    # loop over all images/batches

    #num_of_batches = epochs / batch_size
    #for batch_idx in range(num_of_batches):
    #    images = train_loader()

    total_loss = 0
    correct = 0
    n = len(test_loader.dataset)

    model.eval() # switches off Dropout and makes BatchNorm use running statistics instead of batch statistics. 
    with torch.no_grad(): # with is Python's context manager — it runs setup code before the block and cleanup code after, automatically (gradients automatically re-enabled)
        # gradients disabled here (# don't need backward —> no graph built → faster + less memory)
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)            # forward pass ← get predictions
            loss = criterion(pred, labels)  # compute loss, mean over batch of 128

            total_loss += loss.item()*  images.size(0)
            correct += (pred.argmax(1) ==labels).sum().item()

    return total_loss/n, correct/n

