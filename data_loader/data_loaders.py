from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets.dataset import ImageList

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ImagelistDataLoader(BaseDataLoader):
    '''
    Image List data loading demo
    '''
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        transform = transforms.Compose([
            #transforms.Resize(args.resize),
            #transforms.RandomCrop(args.crop_size),
            transforms.ColorJitter(brightness=0.15, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        #self.root = root
        self.file_dir = data_dir
        self.dataset = ImageList(self.file_dir,transform=transform)
        super(ImagelistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
