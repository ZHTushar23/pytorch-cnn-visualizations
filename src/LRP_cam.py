# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:32:09 2022

@author: ut
"""
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from misc_functions import apply_heatmap, get_example_params
from unet_parts import *
from cbam import * 
class LRP():
    """
        Layer-wise relevance propagation with gamma+epsilon rule

        This code is largely based on the code shared in: https://git.tu-berlin.de/gmontavon/lrp-tutorial
        Some stuff is removed, some stuff is cleaned, and some stuff is re-organized compared to that repository.
    """
    def __init__(self, model):
        self.model = model

    def LRP_forward(self, layer, input_tensor, gamma=None, epsilon=None):
        # This implementation uses both gamma and epsilon rule for all layers
        # The original paper argues that it might be beneficial to sometimes use
        # or not use gamma/epsilon rule depending on the layer location
        # Have a look a the paper and adjust the code according to your needs

        # LRP-Gamma rule
        if gamma is None:
            gamma = lambda value: value + 0.05 * copy.deepcopy(value.data.detach()).clamp(min=0)
        # LRP-Epsilon rule
        if epsilon is None:
            eps = 1e-9
            epsilon = lambda value: value + eps

        # Copy the layer to prevent breaking the graph
        layer = copy.deepcopy(layer)

        # Modify weight and bias with the gamma rule
        try:
            layer.weight = nn.Parameter(gamma(layer.weight))
        except AttributeError:
            pass
            # print('This layer has no weight')
        try:
            layer.bias = nn.Parameter(gamma(layer.bias))
        except AttributeError:
            pass
            # print('This layer has no bias')
        # Forward with gamma + epsilon rule
        return epsilon(layer(input_tensor))
        # return layer(input_tensor)

    def LRP_step(self, forward_output, layer, LRP_next_layer):
        # Enable the gradient flow
        forward_output = forward_output.requires_grad_(True)
        # Get LRP forward out based on the LRP rules
        lrp_rule_forward_out = self.LRP_forward(layer, forward_output, None, None)
        
        # Perform element-wise division
        ele_div = (LRP_next_layer / lrp_rule_forward_out).data
        # Propagate
        (lrp_rule_forward_out * ele_div).sum().backward()
        # print(forward_output.grad)
        # Get the visualization
        LRP_this_layer = (forward_output * forward_output.grad).data

        return LRP_this_layer

    def generate(self, input_image, target_class):
        # layers_in_model = list(self.model._modules['features'])
        
        layers_in_model_init = get_layers(self.model)
        layers_in_model = []
        

        for name, layer in layers_in_model_init:
            if name.endswith("cot") or name.endswith("cer") :
                dd = get_layers_details(layer,name)
                for nm, ll in dd:
                    layers_in_model.append((nm,ll))     
            else:
                layers_in_model.append((name,layer))   
        
        number_of_layers = len(layers_in_model)
        # Forward outputs start with the input image
        forward_output = [input_image]
        forward_output_name = ['input']
        attn_output    = [] 
        # Then we do forward pass with each layer
        for name, layer in layers_in_model:
            # print(name)
            if name=="model.up2":
                forward_output.append(layer.forward(forward_output[-1].detach(),attn_output[-1].detach()))    
                forward_output_name.append(name)   
            elif name=="model.up1":
                forward_output.append(layer.forward(forward_output[-1].detach(),attn_output[-2].detach()))
                forward_output_name.append(name)
            elif name=="model.cer.double_conv.0":
                forward_output.append(layer.forward(forward_output[-7].detach()))
                forward_output_name.append(name)
            elif name=="model.attn1":
                attn_output.append(layer.forward(forward_output[1].detach()))
            elif name=="model.attn2":
                attn_output.append(layer.forward(forward_output[2].detach()))
            else:
                forward_output.append(layer.forward(forward_output[-1].detach()))
                forward_output_name.append(name)

        # print(" Length of Forward Output: ", len(forward_output), len(forward_output_name))
        target_diff_cot = forward_output[-7]-target_class[:,0]
        target_diff_cer = forward_output[-1]-target_class[:,1]
        # This is where we accumulate the LRP results
        # LRP_per_layer = [None] * number_of_layers + [(forward_output[-1] * target_class).data]
        LRP_per_layer = [None] * (number_of_layers-6-6-2) + [None] * 6+[target_diff_cot.data]+ [None] * 6+[target_diff_cer.data]
        # print(len(LRP_per_layer))
        for layer_index in range(1, number_of_layers)[::-1]:
            # This is where features to classifier change happens
            # Have to flatten the lrp of the next layer to match the dimensions
            # if isinstance(layers_in_model[layer_index][1], (nn.Conv2d, nn.ReLU, nn.BatchNorm2d,nn.ConvTranspose2d)):
            # print("LRP Generated for: ",layers_in_model[layer_index][0])
            if layer_index>13:
                if isinstance(layers_in_model[layer_index][1], (nn.Conv2d, nn.ConvTranspose2d)):
                # In the paper implementation, they replace maxpool with avgpool because of certain properties
                # I didn't want to modify the model like the original implementation but
                # feel free to modify this part according to your need(s)
                # forward output is layer_index-2 bcoz attn outputs are not included in forward output.
                    # print("Forward Output Name: ",forward_output_name[layer_index-2])
                    print(layer_index-1)
                    lrp_this_layer = self.LRP_step(forward_output[layer_index-2], layers_in_model[layer_index][1], LRP_per_layer[layer_index])
                    LRP_per_layer[layer_index-1] = lrp_this_layer
                    
                else:
                    LRP_per_layer[layer_index-1] = LRP_per_layer[layer_index]
            # elif layer_index==13:
            #     print("Forward Output Name: ",forward_output_name[layer_index-2-6],forward_output[layer_index-2-6].shape)
            #     # lrp_this_layer = self.LRP_step(forward_output[layer_index-2-6], layers_in_model[layer_index][1], LRP_per_layer[layer_index])
            #     # skipped since up has relu layer
            #     LRP_per_layer[layer_index-1] = lrp_this_layer

            elif layer_index>7 and layer_index<13:
                if isinstance(layers_in_model[layer_index][1], (nn.Conv2d, nn.ConvTranspose2d)):
                    # print("Forward Output Name: ",forward_output_name[layer_index-2])
                    print(layer_index-2)
                    lrp_this_layer = self.LRP_step(forward_output[layer_index-2], layers_in_model[layer_index][1], LRP_per_layer[layer_index-1]) # skip 1 lrp for cer
                    LRP_per_layer[layer_index-2] = lrp_this_layer   
                else:
                    LRP_per_layer[layer_index-2] = LRP_per_layer[layer_index-1]
                #     lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                #     LRP_per_layer[layer_index] = lrp_this_layer

            else:
                LRP_per_layer[layer_index-1] = LRP_per_layer[layer_index]
            
        return LRP_per_layer

def get_layers(model):
    conv2d_layers = []

    def traverse_model(module, name):
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, (DoubleConv, Down, Up, CBAM)):
                conv2d_layers.append((name + '.' + sub_name, sub_module))
            elif isinstance(sub_module, nn.Module):
                traverse_model(sub_module, name + '.' + sub_name)

    traverse_model(model, 'model')
    return conv2d_layers

def get_layers_details(model,cc='model'):
    conv2d_layers = []

    def traverse_model(module, name):
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, (nn.Conv2d, nn.ReLU, nn.BatchNorm2d,nn.ConvTranspose2d, CBAM)):
                conv2d_layers.append((name + '.' + sub_name, sub_module))
            elif isinstance(sub_module, nn.Module):
                traverse_model(sub_module, name + '.' + sub_name)

    traverse_model(model, cc)
    return conv2d_layers


if __name__ == '__main__':
    from cbam import * 
    from unet_parts import *
    from cam9 import CAM9
    # # Get params
    # target_example = 2  # Spider
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)

    prep_img = torch.rand(1, 2, 64, 64)
    target_class = torch.rand(1, 2, 64, 64)
    pretrained_model = CAM9(in_channels=2,gate_channels=64)
    # LRP
    layerwise_relevance = LRP(pretrained_model)

    # Generate visualization(s)
    LRP_per_layer = layerwise_relevance.generate(prep_img, target_class)

    # # Convert the output nicely, selecting the first layer
    # lrp_to_vis = np.array(LRP_per_layer[1][0]).sum(axis=0)
    # lrp_to_vis = np.array(Image.fromarray(lrp_to_vis).resize((prep_img.shape[2],
    #                       prep_img.shape[3]), Image.ANTIALIAS))

    # # Apply heatmap and save
    # heatmap = apply_heatmap(lrp_to_vis, 4, 4)
    # heatmap.figure.savefig('../results/LRP_out.png')


    # for name, layer in conv2d_layers_with_names:
    #     print(name)



    # # Specified index order
    # new_index_order = [0, 3, 1, 4, 2, 5,6,7,8]

    # # Rearrange the elements based on the new index order
    # model_layers = [conv2d_layers_with_names[i] for i in new_index_order]     

    #--------------------+++++++++++++++++++++++++++++++++++++++++++++++++----------------------------------
    # model = CAM9(in_channels=2,gate_channels=64)
    # conv2d_layers_with_names = get_layers(model)
    # x = torch.rand(5, 2, 64, 64)

    # forward_output = [x] 
    # attn_output    = []  
    # attn_count=0
    # layer_count=0

    # for name, layer in conv2d_layers_with_names:
    #     print(name)
    #     if name=="model.up2":
    #         forward_output.append(layer.forward(forward_output[-1].detach(),attn_output[-1].detach()))       
    #     elif name=="model.up1":
    #         forward_output.append(layer.forward(forward_output[-1].detach(),attn_output[-2].detach()))
    #     elif name=="model.cer":
    #         forward_output.append(layer.forward(forward_output[-2].detach()))
    #     elif name=="model.attn1":
    #         attn_output.append(layer.forward(forward_output[1].detach()))
    #     elif name=="model.attn2":
    #         attn_output.append(layer.forward(forward_output[2].detach()))
    #     else:
    #         forward_output.append(layer.forward(forward_output[-1].detach()))

    # target_diff = torch.cat((forward_output[-2],forward_output[-1]),dim=1)-x
    # print(target_diff.data.shape)
   #--------------------+++++++++++++++++++++++++++++++++++++++++++++++++----------------------------------
    # only cot and cer
    # model = CAM9(in_channels=2,gate_channels=64)
    # conv2d_layers_with_names = get_layers(model)
    # model_layers_with_names = []
    # x = torch.rand(5, 2, 64, 64)


    # forward_output = [x] 
    # attn_output    = []  
    # attn_count=0
    # layer_count=0
    # for name, layer in conv2d_layers_with_names:
    #     if name.endswith("cot") or name.endswith("cer") :
    #         dd = get_layers_details(layer,name)
    #         for nm, ll in dd:
    #             model_layers_with_names.append((nm,ll))     
    #     else:
    #         model_layers_with_names.append((name,layer))      
    # for name, layer in model_layers_with_names:
    #     print(name)
    #     if name=="model.up2":
    #         forward_output.append(layer.forward(forward_output[-1].detach(),attn_output[-1].detach()))       
    #     elif name=="model.up1":
    #         forward_output.append(layer.forward(forward_output[-1].detach(),attn_output[-2].detach()))
    #     elif name=="model.cer.double_conv.0":
    #         forward_output.append(layer.forward(forward_output[-7].detach()))
    #     elif name=="model.attn1":
    #         attn_output.append(layer.forward(forward_output[1].detach()))
    #     elif name=="model.attn2":
    #         attn_output.append(layer.forward(forward_output[2].detach()))
    #     else:
    #         forward_output.append(layer.forward(forward_output[-1].detach()))
    # target_diff = torch.cat((forward_output[-7],forward_output[-1]),dim=1)-x
    # print(target_diff.data.shape)

    # target_diff1 = forward_output[-7]-target_class[:,0]
    # target_diff2 = forward_output[-1]-target_class[:,1]

    # print(target_diff1.data.shape)
    # print(target_diff2.data.shape)

    # for layer_index in range(1, len(model_layers_with_names))[::-1]:
    #     print(layer_index)
    #     print(model_layers_with_names[layer_index][0])
    #--------------------+++++++++++++++++++++++++++++++++++++++++++++++++----------------------------------
    # model = CAM9(in_channels=2,gate_channels=64)
    # model_layers_with_names = get_layers_details(model)
    # for name, layer in model_layers_with_names:
    #     print(name)