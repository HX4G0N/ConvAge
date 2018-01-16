import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.serialization import load_lua
from torch.legacy.nn import SpatialConvolution
from classifiers.conv_age import ConvAge

'''
#start: need to download .t7 file
import torch.utils.model_zoo as model_zoo
model_urls = {
    'vgg_face': 'http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz'
}

def vgg_face():
    model = model_zoo.load_url(model_urls['vgg_face'])
    return model
'''

image = Image.open('ak.png').convert('RGB')
image = np.asarray(image)
image = np.reshape(image,(3,224,224))
image = torch.from_numpy(image).float()

#Local implementation
vgg_face_t7 = load_lua('classifiers/VGG_FACE.t7', unknown_classes=True)
layer0 = SpatialConvolution(3 , 64, 3, 3, dW=1, dH=1, padW=1, padH=None)
layer1 = SpatialConvolution(27, 64, 3, 3, dW=1, dH=1, padW=1, padH=None)

#print('layer0 weights size:', layer0.weight.size())
#print('loaded weights size:', vgg_face_t7.modules[0].weight.size())
weights0 = vgg_face_t7.modules[0].weight
weights1 = vgg_face_t7.modules[2].weight
layer0.weight = weights0.resize_(3 , 64, 3, 3)
layer1.weight = weights1.resize_(64, 64, 3, 3)

print(weights0.size())
print(weights1.size())

out0 = layer0.forward(image)
out1 = nn.ReLU(out0)
#out2 = layer1.forward(out1) #size of input problem
print(out0.size())

#Loading from classifier
model = ConvAge()
out = model.forward(image)
