"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import torch.nn as nn

from misc_functions import get_example_params, convert_to_grayscale, save_gradient_images
from nataraja_unet import Nataraja

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        #### Register hook to the first layer

        #1. # Get Conv2d layers with names
        conv2d_layers_with_names = get_conv2d_layers_with_names(self.model)    
        
        #2. separate first layer
        first_layer = conv2d_layers_with_names[0][1]
        # filters = first_layer.weight.data
        # num_filters = filters.size(0)
        # print(num_filters)
        # for i, filter in enumerate(filters):
        #     print(filter.shape)
        # print("Filters: ",i)


        # first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Set requires_grad to True for input image
        input_image.requires_grad = True
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=torch.ones_like(target_class))
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = input_image.grad.data.numpy()
        return gradients_as_arr

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
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    
    # load Model
    pretrained_model = Nataraja(n_channels=2,n_classes=2)    
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # # Generate gradients
    # # print(original_image.size)
    # # print(prep_img.shape)
    # vanilla_grads = VBP.generate_gradients(torch.unsqueeze(prep_img[:,0,:,:],0), torch.unsqueeze(prep_img[:,0,:,:],0))
    # # # Save colored gradients
    # # save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # # Convert to grayscale
    # grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # # Save grayscale gradients
    # save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    # print('Vanilla backprop completed')
