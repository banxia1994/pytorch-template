# by wei
import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pdb

ImageFile.LOAD_TRUNCATED_IAMGES = True


def PIL_loader(path):
    try:
        img = Image.open(path).convert('RGB')
        # print(path)
    except IOError:
        print('Cannot load image ' + path)
    else:
        return img


## for training #
def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


##
class VideoImageList(data.Dataset):
    '''
     Args:
        root (string): Root directory path.
        fileList (string): Image list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    '''

    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader, test_flag=False):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader
        self.test_flag = test_flag

    def __getitem__(self, index):

        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        ## for testing
        if self.test_flag:
            video_name = os.path.join(self.root, imgPath).split('/')[-2]
            return img, target, video_name
        else:
            return img, target, 'None'

    def __len__(self):
        return len(self.imgList)


##
class ImageList(data.Dataset):
    '''
     Args:
        root (string): Root directory path.
        fileList (string): Image list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    '''

    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader, test_flag=False):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader
        self.test_flag = test_flag

    def __getitem__(self, index):

        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        ## for testing
        if self.test_flag:
            img_name = os.path.join(self.root, imgPath).split('/')[-1]
            return img, target, img_name
        else:
            return img, target, 'None'

    def __len__(self):
        return len(self.imgList)
