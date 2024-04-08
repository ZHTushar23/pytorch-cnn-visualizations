import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from misc_functions import apply_heatmap, get_example_params

class LRP():
    def __init__(self, model):
        self.model = model
        self.encoder_layers = list(model._modules['conv1']) + list(model._modules['conv2']) + \
                              list(model._modules['conv3']) + list(model._modules['conv4']) + \
                              list(model._modules['pool']) + list(model._modules['conv5'])
        self.decoder_layers = list(model._modules['upconv4']) + list(model._modules['conv6']) + \
                              list(model._modules['upconv3']) + list(model._modules['conv7']) + \
                              list(model._modules['upconv2']) + list(model._modules['conv8']) + \
                              list(model._modules['upconv1']) + list(model._modules['conv9']) + \
                              list(model._modules['conv10'])

def get_layers_with_names(model):
    conv2d_layers = []

    def traverse_model(module, name):
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, (nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d,nn.ReLU, CBAM)):
                conv2d_layers.append((name + '.' + sub_name, sub_module))
            elif isinstance(sub_module, nn.Module):
                traverse_model(sub_module, name + '.' + sub_name)

    traverse_model(model, 'model')
    return conv2d_layers

if __name__ == '__main__':

    from cam9 import CAM9
    from unet_parts import *
    from cbam import *
    model = CAM9(in_channels=3,gate_channels=64)
    conv2d_layers_with_names = get_layers_with_names(model)  
    i=0
    for name, layer in conv2d_layers_with_names:
        i+=1
        # print(name)
        # if isinstance(layer,Down):
        if name.startswith("model.inc"):
            print(name)
    print("total layer: ",i)