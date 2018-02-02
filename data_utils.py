# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:20:32 2018

@author: Monkey-PC
"""

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
            self.image_folders = f.read().splitlines()

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

        grayscale = transforms.Grayscale()
        #norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.5, 0.5, 0.5])
        topil = transforms.ToPILImage()

        #img = to_tensor(img)
        #img = norm(img)
        #img = topil(img)
        #img = grayscale(img)
        img = to_tensor(img)
        #img = torch.squeeze(img)

        target = Image.open(os.path.join(self.root_dir_name,
                                        img_folder,
                                         '1.png'))
        #target = to_tensor(target)
        #target = norm(target)
        #target = topil(target)
        #target = grayscale(target)
        target = to_tensor(target)
        target = torch.squeeze(target)
        #target *= 255
        #target = np.floor(target)
        #target = torch.from_numpy(target)

        return img, target

class FaceDataCropped(data.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_folders = f.read().splitlines()

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
        grayscale = transforms.Grayscale()

        img = Image.open(os.path.join(self.root_dir_name,
                                      img_folder, 'y/1.png'))
        img_dat = np.array(img)
        img_dat[:,:,3] = img_dat[:,:,3]>0
        img_dat[:,:,0] = img_dat[:,:,0] * img_dat[:,:,3]
        img_dat[:,:,1] = img_dat[:,:,1] * img_dat[:,:,3]
        img_dat[:,:,2] = img_dat[:,:,2] * img_dat[:,:,3]
        img_dat[:,:,3] = img_dat[:,:,3] * 255
        img = Image.fromarray(img_dat).convert('RGB')
        #norm = transforms.Normalize(mean=[0, 0, 0],
        #                     std=[1, 1, 1])
        #img = norm(img)
        #img = grayscale(img)
        img = to_tensor(img)
        #img *= 255

        target = Image.open(os.path.join(self.root_dir_name,
                                        img_folder,
                                         '1.png'))
        targ_dat = np.array(target)
        targ_dat[:,:,3] = targ_dat[:,:,3]>0
        targ_dat[:,:,0] = targ_dat[:,:,0] * targ_dat[:,:,3]
        targ_dat[:,:,1] = targ_dat[:,:,1] * targ_dat[:,:,3]
        targ_dat[:,:,2] = targ_dat[:,:,2] * targ_dat[:,:,3]
        targ_dat[:,:,3] = targ_dat[:,:,3] * 255
        target = Image.fromarray(targ_dat).convert('RGB')
        #target = norm(target)
        target = grayscale(target)
        target = to_tensor(target)
        target = torch.squeeze(target)
        #target *= 255
        #target = np.floor(target)
        #target = torch.from_numpy(target)

        return img, target
