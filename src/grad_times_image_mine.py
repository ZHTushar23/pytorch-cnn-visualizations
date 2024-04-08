"""
Created on Wed Jun 19 17:12:04 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from vanilla_backprop_reg import VanillaBackprop
# from guided_backprop import GuidedBackprop  # To use with guided backprop
# from integrated_gradients import IntegratedGradients  # To use with integrated grads

from nataraja_unet import Nataraja
from v71_dataloader import NasaDataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from visualization import *
from cam9 import CAM9

if __name__ == '__main__':
    # Get params and Dataset
    filename = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/v71_saved_model/nataraja/nataraja_fold_4_20240120_203151.pth"
    # target_example = 0  # Snake
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)

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
    # for i in range(len(loader.dataset)):
        # data = loader.dataset[i]
   
        # # get the data
        # r_test, m_test = data['rad_patches'],data['cot_patches']
        # # Iterate through the list in batches
        # for p_b in range(0, len(r_test), p_batch_size):
        # # for p_b in range(0,len(r_test)):
        #     tensor_list1= r_test[p_b:p_b+p_batch_size]
        #     X_test = torch.stack(tensor_list1, dim=0)
        #     tensor_list2 = m_test[p_b:p_b+p_batch_size]
        #     Y_test = torch.stack(tensor_list2, dim=0)

    data = loader.dataset[10]
    r_test, m_test = data['rad_patches'],data['cot_patches']

    X_test = r_test[0]
    Y_test = m_test[0]
    X_test = torch.unsqueeze(X_test,0)
    Y_test = torch.unsqueeze(Y_test,0)

    X_test = X_test.to(dtype=torch.float)
    Y_test = Y_test.to(dtype=torch.float)
    print("ip Shape: ",X_test.shape)
    print(" op Shape: ",Y_test.shape)
    # load Model
    # pretrained_model = Nataraja(n_channels=2,n_classes=2)  
    pretrained_model = CAM9(in_channels=2,gate_channels=64)
    filename = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/v71_saved_model/cam9/cam9_fold_4_20240123_011709.pth"

    pretrained_model.load_state_dict(torch.load(filename,map_location=torch.device('cpu')))
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(X_test,Y_test)
    # print(type(vanilla_grads), vanilla_grads.shape)

    # Normalize
    vanilla_grads = vanilla_grads - vanilla_grads.min()
    vanilla_grads /= vanilla_grads.max()
    # Make sure dimensions add up!
    im = X_test.detach().numpy()[0]
    grad_times_image = vanilla_grads * im

    fname = "dummy.png"
    plot_cot2(grad_times_image[0,0],"Gradient Radiance 0.66um",fname, False,[0,2])

    fname = "dummy2.png"
    plot_cot2(grad_times_image[0,1],"Gradient Radiance 0.87um",fname, False,[0,2])

    fname = "dummy_g.png"
    plot_cot2(vanilla_grads[0,0],"Gradient .66um",fname, False,[0,2])

    fname = "dummy_g2.png"
    plot_cot2(vanilla_grads[0,1],"Gradient 0.87um",fname, False,[0,2])


    fname = "dummy_t.png"
    plot_cot2(im[0],"True Radiance 0.66um",fname, False,[0,2])

    fname = "dummy_t2.png"
    plot_cot2(im[1],"True Radiance 0.87um",fname, False,[0,2])

    # Savefig
    print('Grad times image completed.')
