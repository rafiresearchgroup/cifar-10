import torch.nn as nn
import torch.nn.functional as F

class CIFAR10NetShallow(nn.Module):
    def __init__(self, cfg ):


        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels= 32, out_channels = M, kernel_size = 5, stride=2)
        self.features = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size, padding=1),
            nn.BatchNorm2d(cfg.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32×32 → 16×16

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.out_channels*16*16, cfg.num_classes)
            # nn.ReLU() no activation
        )
    

    def forward(self, x):
        features = self.features(x)
        pred = self.classifier(features)
        return pred






class CIFAR10NetDeeper(nn.Module):
    def __init__(self, cfg ):


        # Layer 1: edges, corners        (low level)
        # Layer 2: textures, shapes      (mid level)  
        # Layer 3: parts, objects        (high level)
        # = richer feature hierarchy
        # = more "understanding" of the image

        # block 1: 3  → 32  channels, 32×32 → 16×16
        # block 2: 32 → 64  channels, 16×16 → 8×8
        # block 3: 64 → 128 channels, 8×8  → 4×4

        # Before the FC:
        # 128 × 4 × 4 = 2048 → Linear(2048, 10)

        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels= 32, out_channels = M, kernel_size = 5, stride=2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, cfg.kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32×32 → 16×16

            nn.Conv2d(32, 64, cfg.kernel_size, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16×16 → 8x8 

            nn.Conv2d(64, 128, cfg.kernel_size, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8 → 4x4 


        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 512), # the network has to compress 2048 features into 10 classes in one step, which is hard to learn. Adding a hidden layer with 512 units:
            nn.ReLU(),
            nn.Dropout(0.5), # During each forward pass, randomly zeros 50% of neurons (regularization)
            nn.Linear(512, cfg.num_classes)

        )
    

    def forward(self, x):
        features = self.features(x)
        pred = self.classifier(features)
        return pred


# remarks

# Conv2d → BatchNorm2d → ReLU
# Dropout comes later in the FC head, not in conv blocks. BatchNorm is important
# MaxPool2d(2) halves spatial dimensions:
# 32×32 → MaxPool → 16×16 → MaxPool → 8×8 → MaxPool → 4×4
# nn.Flatten()  # (batch, 256, 4, 4) → (batch, 256*4*4) = (batch, 4096)

if __name__ == "__main__":
    import torch
    model = CIFAR10NetShallow()

    x = torch.randn(4, 3, 32, 32)  # fake batch of 4 images
    out = model(x)
    print(out.shape)  # should be (4, 10)
