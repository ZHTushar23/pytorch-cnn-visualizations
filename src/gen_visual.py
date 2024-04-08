
from LRP_cam import *
from v71_dataloader import NasaDataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from visualization import *
from cam9 import CAM9
from matplotlib import pyplot as plt

# Get params and Dataset
# filename = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/v71_saved_model/nataraja/nataraja_fold_4_20240120_203151.pth"
pretrained_model = CAM9(in_channels=2,gate_channels=64)
filename = "/home/ztushar1/psanjay_user/COT_CER_Joint_Retrievals/v71_saved_model/cam9/cam9_fold_4_20240123_011709.pth"
pretrained_model.load_state_dict(torch.load(filename,map_location=torch.device('cpu')))

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
                        patch_size=64,stride=20, cp=False)
print(len(test_data))
loader = DataLoader(test_data, batch_size=1,shuffle=False)
dir_namme = "../results/LRP_CAM/"
for kk in range (180):
    data = loader.dataset[kk]
    p_num = data["p_num"]
    if p_num in [208,327,33,0,332]:
        r_test, m_test = data['rad_patches'],data['cot_patches']
        rn_test = data['rad_patches_unnorm']
        tx_func = data['tx_func']
    else:
        continue

    for pp in range(len(r_test)):
        if pp%1==0:
            X_test = r_test[pp]
            Y_test = m_test[pp]
            X_test = torch.unsqueeze(X_test,0)
            Y_test = torch.unsqueeze(Y_test,0)
            x_unnorm = rn_test[pp]
            x_unnorm = torch.unsqueeze(x_unnorm,0)

            X_test = X_test.to(dtype=torch.float)
            Y_test = Y_test.to(dtype=torch.float)
            print("ip Shape: ",X_test.shape, x_unnorm.shape)
            print(" op Shape: ",Y_test.shape)

            # mask = torch.ones(1,2,64, 64)
            # # Set a 2x2 region to 1
            # mask[0,:,58:62,16:20] = 0
            # mask = mask.to(dtype=torch.float)
            # print(" Mask Shape: ",mask.shape, torch.sum(mask))

            # # Apply Mask
            # Y_test = Y_test*mask 
            # # X_test = X_test*mask
            # x_unnorm = x_unnorm*mask            

            # Perform LRP
            layerwise_relevance = LRP(pretrained_model)
            # Generate visualization(s)
            LRP_per_layer = layerwise_relevance.generate(X_test, Y_test)
            lrp_to_vis_cer = np.array(LRP_per_layer[15][0]).sum(axis=0)
            lrp_to_vis_cot = np.array(LRP_per_layer[15][0]).sum(axis=0)
            heatmap_cot = apply_heatmap(lrp_to_vis_cot, 4, 4)
            heatmap_cer = apply_heatmap(lrp_to_vis_cer, 4, 4)

            # Rad

            fname = dir_namme+"rad066_profile_%05d_patch_%03d.png"%(p_num,pp)
            plot_cot2(x_unnorm[0,0],"Radiance at 0.66um",fname, False,[0,2])  
            fname = dir_namme+"rad213_profile_%05d_patch_%03d.png"%(p_num,pp)
            plot_cot2(x_unnorm[0,1],"Radiance at 2.13um",fname, False,[0,2])  

            # fname = dir_namme+"mask_profile_%05d_patch_%03d.png"%(p_num,pp)
            # plot_cmask4(mask[0,0],"Mask",fname)   

      
            # COT
            fname = dir_namme+"cot_profile_%05d_patch_%03d.png"%(p_num,pp)
            plot_cot2(Y_test[0,0],"True COT",fname, False,[0,7])
            # CER
            fname = dir_namme+"cer_profile_%05d_patch_%03d.png"%(p_num,pp)
            plot_cot2(Y_test[0,1]*30,"True CER",fname, False,[0,40])            

            

            # LRP COT
            fname = dir_namme+"cot_lrp_profile_%05d_patch_%03d_gistgray.png"%(p_num,pp)
            # plot_cot2(lrp_to_vis_cot,"LRP COT",fname, False,[0,1])
            plot_cmask4(lrp_to_vis_cot,"LRP COT",fname,[0,.5])
            # LRP CER
            fname = dir_namme+"cer_lrp_profile_%05d_patch_%03d_gist_gray.png"%(p_num,pp)
            # plot_cot2(lrp_to_vis_cer,"LRP CER",fname, False,[0,1]) 
            plot_cmask4(lrp_to_vis_cer,"LRP CER",fname,[0,.5])

            # LRP COT
            fname = dir_namme+"cot_lrp_profile_%05d_patch_%03d_plasma.png"%(p_num,pp)
            # plot_cot2(lrp_to_vis_cot,"LRP COT",fname, False,[0,1])
            plot_cmask3(lrp_to_vis_cot,"LRP COT",fname)
            # LRP CER
            fname = dir_namme+"cer_lrp_profile_%05d_patch_%03d_plasma.png"%(p_num,pp)
            # plot_cot2(lrp_to_vis_cer,"LRP CER",fname, False,[0,1]) 
            plot_cmask3(lrp_to_vis_cer,"LRP CER",fname)


            # Heatmaps
            fname = dir_namme+"heatmap_cot_profile_%05d_patch_%03d.png"%(p_num,pp)
            heatmap_cot.figure.savefig(fname)
            plt.close()
            
            fname = dir_namme+"heatmap_cer_profile_%05d_patch_%03d.png"%(p_num,pp)
            heatmap_cer.figure.savefig(fname)
            plt.close()
            
            
            # print(np.min(lrp_to_vis_cot),np.max(lrp_to_vis_cot))
            # print(np.min(lrp_to_vis_cer),np.max(lrp_to_vis_cer))
            # for layer_index in range(5, len(LRP_per_layer))[::-1]:
            #     lrp_to_vis = np.array(LRP_per_layer[layer_index][0]).sum(axis=0)
            #     # print(lrp_to_vis.shape)
            #     break
            #     # fname = dir_namme+"out_profile_%05d_patch_%03d.png"%(p_num,pp)
            #     # plot_cot2(Y_test[0,0],"True COT",fname, False,[0,7])