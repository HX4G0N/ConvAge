"""ConvAge"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.serialization import load_lua
from torch.legacy.nn import SpatialConvolution


class ConvAge(nn.Module):

    def __init__(self, init_weights=True):
        super(ConvAge, self).__init__()
        self.Spatial0 = SpatialConvolution(3, 64, 3, 3, dW=1, dH=1, padW=1, padH=None)
        self.Spatial1 = SpatialConvolution(64, 64, 3, 3, dW=1, dH=1, padW=1, padH=None)

        if init_weights:
            self._initialize_weights()

    def forward(self, inp):
        x = self.Spatial0.forward(inp)
        x = nn.ReLU(x)
        #x = self.Spatial1.forward(x)
        #x = nn.ReLU(x)
        return x

    def _initialize_weights(self):
        path = 'classifiers/VGG_FACE.t7'
        vgg_face_t7  = load_lua(path, unknown_classes=True)
        weights0 = vgg_face_t7.modules[0].weight
        weights1 = vgg_face_t7.modules[2].weight

        self.Spatial0.weight = weights0.resize_(3, 64, 3, 3)
        self.Spatial1.weight = weights1.resize_(64, 64, 3, 3)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cudag
