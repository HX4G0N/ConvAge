import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from solver import Solver
from data_utils import FaceData
from torch.utils.serialization import load_lua
from classifiers.conv_age import SegmentationNN


train_data = FaceData(image_paths_file='LAG/train/train.txt')
val_data = FaceData(image_paths_file='LAG/val/val.txt')
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=1)

val_loader = torch.utils.data.DataLoader(val_data,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=1)
#Model loading
model = SegmentationNN()

solver = Solver(optim_args={"lr": 1e-4,
                            "eps": 1e-8
                            },
                loss_func = torch.nn.CrossEntropyLoss(ignore_index = -1))

solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=1)
