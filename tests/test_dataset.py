from src.dataset import get_data_loader
from src.config import Config

# read data


def test_dataset_batch_shape():
    cfg = Config()
    
    train_loader, test_loader = get_data_loader(cfg)

    images,labels = next(iter(train_loader))
    train_loader_shape = images.shape

    images,labels = next(iter(test_loader))
    test_loader_shape = images.shape

    assert labels.shape == (cfg.batch_size,)  # ← add this


    assert train_loader_shape == (cfg.batch_size, 3, 32, 32)
    assert test_loader_shape == (cfg.batch_size, 3, 32, 32)



def test_dataset_size():
    cfg = Config()
    
    train_loader, test_loader = get_data_loader(cfg)
    assert len(train_loader.dataset) == 50000
    assert len(test_loader.dataset) == 10000


def test_labels_range():
    cfg = Config()
    train_loader, _ = get_data_loader(cfg)
    _, labels = next(iter(train_loader))
    assert labels.min() >= 0
    assert labels.max() <= 9

def test_images_normalized():
    cfg = Config()
    train_loader, _ = get_data_loader(cfg)
    images, _ = next(iter(train_loader))
    assert images.min() < 0      # normalized, not raw [0,255]
    assert images.max() < 10     # not raw pixels


def test_images_mean_sdt():

    cfg = Config()    
    train_loader, test_loader = get_data_loader(cfg)

    # breakpoint()
    
    images,labels = next(iter(train_loader))
    image_mean = images.mean(dim=(1,2,3))

    assert abs(image_mean) < 0.2

