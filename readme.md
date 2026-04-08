https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html



cifar10/
├── dataset.py      # data loading, transforms, DataLoader
├── model.py        # CNN architecture (nn.Module)
├── train.py        # train loop + val loop
├── evaluate.py     # final test evaluation + confusion matrix
└── main.py         # wires everything together, hyperparams



Epoch 1:
  batch 1   (images 0-127)    → forward → loss → backward → update weights
  batch 2   (images 128-255)  → forward → loss → backward → update weights
  ...
  batch 390 (images 49872-49999) → forward → loss → backward → update weights
← one full epoch done, weights updated 390 times
  
Epoch 2:
  (same 50,000 images, reshuffled)


  One consequence: with epochs=50 and batch_size=128 your model will see 50 × 390 = 19,500 weight updates total before training is done.



CIFAR training (25 epochs):  Train 87% | Test 83%  ✅
MNIST fine-tuning (25 epochs): Train 98% | Test 98.2% ✅
CIFAR after fine-tuning:     Test 15.6%  ← expected, model forgot CIFAR


Random init:      90.4% already at epoch 0!  ← faster bootstrap
CIFAR pretrained: 47.6% at epoch 0           ← slower start
Random final:     98.9%                       ← slightly higher
CIFAR final:      98.2%                       ← slightly lower


models:
-------

ShallowNet (1 block, 100 epochs): Train 58% | Test 63%

DeeperNet:
Train keeps improving, test plateaus at ~84% from epoch 35 onwards. Classic overfitting signal.
Epoch: 00/50 || Train Loss: 1.5780 | Train Acc: 41.3% ||  Test Loss: 1.3049 | Test Acc: 53.3%
Epoch: 01/50 || Train Loss: 1.1844 | Train Acc: 57.2% ||  Test Loss: 0.9837 | Test Acc: 64.7%
Epoch: 02/50 || Train Loss: 0.9644 | Train Acc: 65.8% ||  Test Loss: 0.8385 | Test Acc: 70.9%
Epoch: 03/50 || Train Loss: 0.8492 | Train Acc: 70.0% ||  Test Loss: 0.7658 | Test Acc: 72.8%
Epoch: 04/50 || Train Loss: 0.7774 | Train Acc: 72.6% ||  Test Loss: 0.6969 | Test Acc: 75.6%
Epoch: 05/50 || Train Loss: 0.7247 | Train Acc: 74.4% ||  Test Loss: 0.6656 | Test Acc: 77.0%
Epoch: 06/50 || Train Loss: 0.6774 | Train Acc: 76.2% ||  Test Loss: 0.6374 | Test Acc: 77.9%
Epoch: 07/50 || Train Loss: 0.6467 | Train Acc: 77.5% ||  Test Loss: 0.6568 | Test Acc: 77.8%
Epoch: 08/50 || Train Loss: 0.6093 | Train Acc: 78.8% ||  Test Loss: 0.6354 | Test Acc: 78.3%
Epoch: 09/50 || Train Loss: 0.5806 | Train Acc: 79.7% ||  Test Loss: 0.5981 | Test Acc: 79.7%
Epoch: 10/50 || Train Loss: 0.5594 | Train Acc: 80.4% ||  Test Loss: 0.6018 | Test Acc: 79.5%
Epoch: 11/50 || Train Loss: 0.5409 | Train Acc: 81.0% ||  Test Loss: 0.6024 | Test Acc: 78.8%
Epoch: 12/50 || Train Loss: 0.5126 | Train Acc: 82.0% ||  Test Loss: 0.5535 | Test Acc: 81.6%
Epoch: 13/50 || Train Loss: 0.5012 | Train Acc: 82.4% ||  Test Loss: 0.5752 | Test Acc: 81.2%
Epoch: 14/50 || Train Loss: 0.4802 | Train Acc: 83.2% ||  Test Loss: 0.5897 | Test Acc: 80.8%
Epoch: 15/50 || Train Loss: 0.4685 | Train Acc: 83.7% ||  Test Loss: 0.5465 | Test Acc: 81.8%
Epoch: 16/50 || Train Loss: 0.4504 | Train Acc: 84.3% ||  Test Loss: 0.5443 | Test Acc: 81.6%
Epoch: 17/50 || Train Loss: 0.4387 | Train Acc: 84.6% ||  Test Loss: 0.5566 | Test Acc: 81.5%
Epoch: 18/50 || Train Loss: 0.4251 | Train Acc: 85.1% ||  Test Loss: 0.5769 | Test Acc: 80.5%
Epoch: 19/50 || Train Loss: 0.4108 | Train Acc: 85.5% ||  Test Loss: 0.5445 | Test Acc: 82.5%
Epoch: 20/50 || Train Loss: 0.4024 | Train Acc: 86.0% ||  Test Loss: 0.5260 | Test Acc: 82.5%
Epoch: 21/50 || Train Loss: 0.3852 | Train Acc: 86.4% ||  Test Loss: 0.5292 | Test Acc: 83.0%
Epoch: 22/50 || Train Loss: 0.3834 | Train Acc: 86.5% ||  Test Loss: 0.5483 | Test Acc: 82.0%
Epoch: 23/50 || Train Loss: 0.3708 | Train Acc: 87.0% ||  Test Loss: 0.5084 | Test Acc: 83.6%
Epoch: 24/50 || Train Loss: 0.3597 | Train Acc: 87.4% ||  Test Loss: 0.5162 | Test Acc: 83.4%
Epoch: 25/50 || Train Loss: 0.3565 | Train Acc: 87.4% ||  Test Loss: 0.5250 | Test Acc: 82.7%
Epoch: 26/50 || Train Loss: 0.3429 | Train Acc: 87.9% ||  Test Loss: 0.5300 | Test Acc: 83.5%
Epoch: 27/50 || Train Loss: 0.3366 | Train Acc: 88.2% ||  Test Loss: 0.5135 | Test Acc: 83.9%
Epoch: 28/50 || Train Loss: 0.3243 | Train Acc: 88.5% ||  Test Loss: 0.5093 | Test Acc: 83.6%
Epoch: 29/50 || Train Loss: 0.3189 | Train Acc: 88.8% ||  Test Loss: 0.5487 | Test Acc: 82.7%
Epoch: 30/50 || Train Loss: 0.3118 | Train Acc: 89.1% ||  Test Loss: 0.5451 | Test Acc: 83.1%
Epoch: 31/50 || Train Loss: 0.3039 | Train Acc: 89.3% ||  Test Loss: 0.5348 | Test Acc: 84.0%
Epoch: 32/50 || Train Loss: 0.2921 | Train Acc: 89.8% ||  Test Loss: 0.5082 | Test Acc: 84.0%
Epoch: 33/50 || Train Loss: 0.2926 | Train Acc: 89.6% ||  Test Loss: 0.5476 | Test Acc: 83.2%
Epoch: 34/50 || Train Loss: 0.2872 | Train Acc: 90.0% ||  Test Loss: 0.5342 | Test Acc: 84.0%
Epoch: 35/50 || Train Loss: 0.2772 | Train Acc: 90.3% ||  Test Loss: 0.5216 | Test Acc: 84.4%
Epoch: 36/50 || Train Loss: 0.2769 | Train Acc: 90.4% ||  Test Loss: 0.5348 | Test Acc: 83.4%
Epoch: 37/50 || Train Loss: 0.2672 | Train Acc: 90.5% ||  Test Loss: 0.5326 | Test Acc: 84.0%
Epoch: 38/50 || Train Loss: 0.2633 | Train Acc: 90.8% ||  Test Loss: 0.5635 | Test Acc: 83.3%
Epoch: 39/50 || Train Loss: 0.2545 | Train Acc: 91.0% ||  Test Loss: 0.5326 | Test Acc: 84.3%
Epoch: 40/50 || Train Loss: 0.2540 | Train Acc: 90.9% ||  Test Loss: 0.5213 | Test Acc: 84.1%
Epoch: 41/50 || Train Loss: 0.2462 | Train Acc: 91.3% ||  Test Loss: 0.5540 | Test Acc: 83.7%
Epoch: 42/50 || Train Loss: 0.2407 | Train Acc: 91.6% ||  Test Loss: 0.5451 | Test Acc: 84.2%
Epoch: 43/50 || Train Loss: 0.2375 | Train Acc: 91.7% ||  Test Loss: 0.5310 | Test Acc: 84.7%
Epoch: 44/50 || Train Loss: 0.2277 | Train Acc: 92.0% ||  Test Loss: 0.5408 | Test Acc: 84.3%
Epoch: 45/50 || Train Loss: 0.2285 | Train Acc: 92.1% ||  Test Loss: 0.5316 | Test Acc: 84.4%
Epoch: 46/50 || Train Loss: 0.2195 | Train Acc: 92.4% ||  Test Loss: 0.5380 | Test Acc: 84.3%
Epoch: 47/50 || Train Loss: 0.2164 | Train Acc: 92.5% ||  Test Loss: 0.5512 | Test Acc: 84.5%
Epoch: 48/50 || Train Loss: 0.2155 | Train Acc: 92.4% ||  Test Loss: 0.5752 | Test Acc: 84.1%
Epoch: 49/50 || Train Loss: 0.2090 | Train Acc: 92.7% ||  Test Loss: 0.5621 | Test Acc: 83.8%