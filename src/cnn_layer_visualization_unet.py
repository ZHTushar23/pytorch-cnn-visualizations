"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image
from nataraja_unet import Nataraja

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (64, 64, 2)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image


            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self,pr_im=None):
        # Process image and return variable
        processed_image = pr_im
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            
            # conv2d_layers_with_names = get_conv2d_layers_with_names(self.model)  
            skip_connections = []
            for index, layer in enumerate(self.model.children()):

                # print(index, layer)
                # if index<5:
                #     # Forward pass layer by layer
                #     x = layer(x)
                #     skip_connections.append(x)
                # else:
                #     # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
                #     y = skip_connections.pop()
                #     x = layer(x,y)  
                if isinstance(layer, Down):
                    print(" Down Layer")
                    # Forward pass layer by layer
                    x = layer(x)
                    if index<4:
                        skip_connections.append(x)
                elif isinstance(layer, DoubleConv):
                    print("d conv")
                    # Forward pass layer by layer
                    x = layer(x)
                    skip_connections.append(x)
                elif isinstance(layer, Up):
                    print(" up")
                    y = skip_connections.pop()
                    # print(y.shape)
                    # print(x.shape)
                    x = layer(x,y)

                else:
                    print("out c")
                    # Forward pass layer by layer
                    x = layer(x)


                if index == self.selected_layer:
                # if name == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
                
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


def get_conv2d_layers_with_names(model):
    conv2d_layers = []

    def traverse_model(module, name):
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Conv2d):
                conv2d_layers.append((name + '.' + sub_name, sub_module))
            elif isinstance(sub_module, nn.Module):
                traverse_model(sub_module, name + '.' + sub_name)

    traverse_model(model, 'model')
    return conv2d_layers

if __name__ == '__main__':
    # cnn_layer = 17
    # filter_pos = 10
    # # Fully connected layer is not needed
    # pretrained_model = models.vgg16(pretrained=True).features
    # layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # # # Layer visualization with pytorch hooks
    # # layer_vis.visualise_layer_with_hooks()

    # # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()

    cnn_layer = 9
    filter_pos = 0
    # Fully connected layer is not needed
    import torch.nn as nn
    from unet_parts import *
    pretrained_model = Nataraja(n_channels=3,n_classes=3)
    # for index, layer in enumerate(pretrained_model.children()):
    #     # print(index, layer)
    #     if isinstance(layer, Down):
    #         print("yes")
    #     elif isinstance(layer, DoubleConv):
    #         print("d conv")
    #     elif isinstance(layer, Up):
    #         print(" up")
    #     else:
    #         print("out c")

    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_without_hooks()


    from v71_dataloader import NasaDataset
    from torch.utils.data import Dataset, DataLoader

    fold = 4
    print("checking fold: ", fold)
    dataset_dir1 = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/one_thousand_profiles/Refl"
    dataset_dir2 = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/ncer_fill3"
    # cv_train_list = np.load("Data_split/train_split_100m.npy")
    # cv_valid_list = np.load("Data_split/valid_split_100m.npy")
    cv_test_list = np.load("/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/Data_split/test_split_100m.npy")
    profilelist  = cv_test_list[fold][:,0]

    test_data = NasaDataset(fold=fold,profilelist=profilelist,
                            root_dir1=dataset_dir1,root_dir2=dataset_dir2,
                            patch_size=64,stride=10, cp=False)
    print(len(test_data))
    loader = DataLoader(test_data, batch_size=1,shuffle=False)
    # for i in range(len(test_data)):
    #     data = loader.dataset[i]
    #     r_test, m_test = data['rad_patches'],data['cot_patches']
    #     p_num = data['p_num']

    #     print(p_num,i)
    #     if p_num==26:
    #         break

    data = loader.dataset[39]
    r_test, m_test = data['rad_patches'],data['cot_patches']
    p_num = data['p_num']

    print(p_num)
    X_test = r_test[0]
    Y_test = m_test[0]
    X_test = torch.unsqueeze(X_test,0)
    Y_test = torch.unsqueeze(Y_test,0)

    X_test = X_test.to(dtype=torch.float)
    Y_test = Y_test.to(dtype=torch.float)
    print("ip Shape: ",X_test.shape)
    print(" op Shape: ",Y_test.shape)

    # # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_without_hooks(X_test)