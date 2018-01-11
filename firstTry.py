"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import _pickle as pickle

class FaceData(data.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_folders = f.read().splitlines() #Carlo was here

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_folders)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_folder = self.image_folders[index]
        img = Image.open(os.path.join(self.root_dir_name,
                                      img_folder, 'y/1.png')).convert('RGB')
        #center_crop = transforms.CenterCrop(240)
        #img = center_crop(img)
        img = to_tensor(img)

        target = Image.open(os.path.join(self.root_dir_name,
                                         img_folder, '1.png')).convert('RGB')
        
        target = to_tensor(target)
        


        return img, target
