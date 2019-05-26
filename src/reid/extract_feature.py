import os
import os.path as op
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
from .modeling import build_model
from .misc import preprocess, normalize_torch, make_transforms
from sklearn.preprocessing import normalize
from PIL import Image

class Reid_Extrctor:
    """Base feature extractor class.
    args:
        features: List of features.
    """
    def __init__(self, model_name,model_path,image_size =320):
        model = build_model(model_name, 2000)
        model.cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        self.extractor = model
        self.image_size = image_size
        self.transforms = preprocess(normalize_torch, self.image_size)
        

    def extract(self,img_path,region):
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
        im = img.crop(region)
        input = self.transforms(im)
        input = input.unsqueeze(0)
        with torch.no_grad():
            input = input.cuda()
            after_feat, before_feat = self.extractor(input)
        features = normalize(before_feat.cpu().data.numpy(), norm='l2').astype('float32')
        return features

