
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Transform → Dataset → DataLoader
# Transform = what happens to each image
# Dataset = indexed collection of transformed images
# DataLoader = batched iterator over the dataset

def get_data_loader(cfg):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
        ])

    # torchvision knows the folder structure automatically
    dataset_train = torchvision.datasets.CIFAR10(
        root = cfg.data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    # torchvision knows the folder structure automatically
    dataset_test = torchvision.datasets.CIFAR10(
        root = cfg.data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    num_of_workers = 0
    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,  num_workers=num_of_workers)
    test_loader  = DataLoader(dataset_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=num_of_workers)
    return train_loader, test_loader

    # return dataset_train, dataset_test





def get_data_loader_mnist(cfg):

    # data_dir = /Users/rdallatorre/repos/cifar-10/data'

    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2470, 0.2435, 0.2616)),
    #     ])
    train_transform=transforms.Compose([
        transforms.Pad(2),              # 28×28 → 32×32 first

        transforms.ToTensor(),
        transforms.Normalize((0.1302),
                             (0.3069))
    ])
    
    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2470, 0.2435, 0.2616)),
    #     ])
    test_transform=transforms.Compose([
        transforms.Pad(2),              # 28×28 → 32×32 first
        transforms.ToTensor(),
        transforms.Normalize((0.1302),
                             (0.3069))
    ])

    # torchvision knows the folder structure automatically
    dataset_train = torchvision.datasets.MNIST(
        root = cfg.data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    # torchvision knows the folder structure automatically
    dataset_test = torchvision.datasets.MNIST(
        root = cfg.data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    num_of_workers = 0
    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,  num_workers=num_of_workers)
    test_loader  = DataLoader(dataset_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=num_of_workers)
    return train_loader, test_loader



if __name__ == "__main__":
    from config import Config
    print("load data")
    train_loader, test_loader = get_data_loader(Config())
    
    train_loader_mnist, test_loader_mnist = get_data_loader_mnist(Config())
    


    # train_loader, test_loader = get_dataset(data_dir)
    # print("train shape = " , train.data.shape) # train shape =  (50000, 32, 32, 3)
    # print("test shape = " , test.data.shape) # test shape =  (10000, 32, 32, 3)

    # CIFAR
    print("CIFAR")

    images,labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   # (128, 3, 32, 32)
    print(f"Labels shape: {labels.shape}")  # (128,)

    print("data loaded")
    

    #MNIST
    print("MNIST")
    images,labels = next(iter(train_loader_mnist))
    print(f"Batch shape: {images.shape}")   # (128, 3, 32, 32)
    print(f"Labels shape: {labels.shape}")  # (128,)

    print("data loaded")

    


