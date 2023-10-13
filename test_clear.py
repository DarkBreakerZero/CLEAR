import os
from datasets import npz_proj_img_reader_func
import numpy as np
import time
import argparse
from pytools import *
from models import *
from utils import *
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


root_dir = '/Data'

model_name = 'clear_with_gan_mslr'

results_save_dir = './runs/' + model_name + '/test/'
make_dirs(results_save_dir)

epoch =29
model_dir = './runs/' + model_name + '/checkpoints/G/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
checkpoint = torch.load(model_dir)

model = GeneratorCLEAR(chl=32)
model = load_model(model, checkpoint).cuda()
model.eval()

test_cases = ['L067', 'L192'] # An example

for case in test_cases:

    hdct_path = root_dir + '/AAPM/' + case + '/full_1mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol = hdct_vol - 1024

    pred_vol = np.zeros(np.shape(hdct_vol), dtype=np.float32)

    # ldct_vol = read_raw_data_all(root_dir + '/sAAPMCT/' + case + '/noisy_CT_5e4/', w=512, h=512, start_index=0, end_index=-13)

    ldproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/noisy_proj_5e4/', w=576, h=736, start_index=0, end_index=-15)
    ldproj_vol[ldproj_vol<0] = 0

    for slice in range(0, np.size(ldproj_vol, 0), 3):

        if slice + 3 <= np.size(ldproj_vol, 0):

            # hdct_slices = hdct_vol[slice:slice+3, :, :]
            # ldct_slices = ldct_vol[slice:slice+3, :, :]
            ldproj_slices = ldproj_vol[slice:slice+3, :, :]

            ldproj_slices = ldproj_slices[np.newaxis, np.newaxis, ...]

            # print(np.shape(ldproj_slices))

            ldProj = torch.FloatTensor(ldproj_slices).cuda()

            with torch.no_grad():

                proj_net, img_fbp, img_net, proj_re = model(ldProj)
                pred_img = np.squeeze(img_net.data.cpu().numpy())
                pred_vol[slice:slice+3, :, :] = pred_img / 0.02 - 1024
        
        else:

            ldproj_slices = ldproj_vol[np.size(ldproj_vol, 0)-3:, :, :]

            ldproj_slices = ldproj_slices[np.newaxis, np.newaxis, ...]

            ldProj = torch.FloatTensor(ldproj_slices).cuda()

            with torch.no_grad():

                proj_net, img_fbp, img_net, proj_re = model(ldProj)
                pred_img = np.squeeze(img_net.data.cpu().numpy())
                pred_vol[np.size(ldproj_vol, 0)-3:, :, :] = pred_img / 0.02 - 1024
            
    (pred_vol).astype(np.float32).tofile(results_save_dir + case + '_' + model_name + '_E' + str(epoch) + '.raw')