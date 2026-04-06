from dataclasses import dataclass



@dataclass
class Config:
    # data
    data_dir : str = "/Users/rdallatorre/repos/cifar-10/data/"
    batch_size : int =128
    num_workers: int =0

    #model (CIFAR10NetShallow)
    in_channels : int = 3
    out_channels : int = 32
    kernel_size : int = 3
    num_classes : int = 10

    # training
    epochs: int = 25
    lr : float = 0.01
    momentum : float = 0.9


    save_dir : str = "/Users/rdallatorre/repos/cifar-10/trained_models/"