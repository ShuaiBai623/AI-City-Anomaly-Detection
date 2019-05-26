import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .augmentation import *

NB_CLASSES = 128

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)


# IMAGE_SIZE = 224
# IMAGE_SIZE = 299


def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_vflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        VerticalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_rotation(normalize, image_size,angle):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        Rotate(angle),
        transforms.ToTensor(),
        normalize
    ])
def preprocess_hflip_rotation(normalize, image_size,angle):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        Rotate(angle),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_vflip_rotation(normalize, image_size,angle):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        VerticalFlip(),
        Rotate(angle),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])
