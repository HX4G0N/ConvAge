import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.serialization import load_lua

class ConvAgeNN(nn.Module):

    def __init__(self, n_classes=256, in_channels=3, is_unpooling=True, init_weight=True):
        super(ConvAgeNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = VGGDown2(self.in_channels, 64)
        self.down2 = VGGDown2(64, 128)
        self.down3 = segnetDown2(128, 256)
        self.down4 = segnetDown2(256, 512)
        self.down5 = segnetDown2(512, 1024)
        self.down6 = segnetDown2(1024,512)
        self.down7 = segnetDown2(512, 256)
        self.down8 = segnetDown2(256,128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

        # self.up5 = segnetUp3(512, 512)
        # self.up4 = segnetUp3(512, 256)
        # self.up3 = segnetUp3(256, 128)
        # self.up2 = segnetUp2(128, 64)
        # self.up1 = segnetUp2(64, n_classes)

        if init_weight:
            self.init_vggFACE_params()

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        # print(down1.size())
        down2, indices_2, unpool_shape2 = self.down2(down1)
        # print(down2.size())
        down3 = self.down3(down2)
        # print(down3.size())
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        down8 = self.down8(down7)
        up2 = self.up2(down8, indices_2, unpool_shape2)
        # print(up2.size())
        up1 = self.up1(up2, indices_1, unpool_shape1)
        # print(up1.size())

        # up5 = self.up5(down5, indices_5, unpool_shape5)
        # up4 = self.up4(up5, indices_4, unpool_shape4)
        # up3 = self.up3(up4, indices_3, unpool_shape3)
        # up2 = self.up2(up3, indices_2, unpool_shape2)
        # up1 = self.up1(up2, indices_1, unpool_shape1)
        return up1

    def init_vggFACE_params(self):
        vgg_face_t7 = load_lua('classifiers/VGG_FACE.t7', unknown_classes=True)
        vgg_face_layers = [vgg_face_t7.modules[0].weight.resize_(64 , 3  , 3, 3),
                           vgg_face_t7.modules[2].weight.resize_(64 , 64 , 3, 3),
                           vgg_face_t7.modules[5].weight.resize_(128, 64 , 3, 3),
                           vgg_face_t7.modules[7].weight.resize_(128, 128, 3, 3),]

        blocks = [self.down1,
                  self.down2]
        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            units = [conv_block.conv1.cb_unit,
                     conv_block.conv2.cb_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        for l1, l2 in zip(vgg_face_layers, merged_layers):
            if isinstance(l2, nn.Conv2d):
                assert l1.size() == l2.weight.size()
                l2.weight.data = l1
                #assert l1.bias.size() == l2.bias.size() #not loading, bias==0
                #l2.bias.data = l1.bias.data

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)



class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DRelu, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                               nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class VGGDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(VGGDown2, self).__init__()
        self.conv1 = conv2DRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape

class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        #self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        #unpooled_shape = outputs.size()
        #outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        #print(int(in_channels), int(n_filters), k_size,padding, stride, bias)
        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs
