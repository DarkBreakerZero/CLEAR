import numpy as np
import os
from pytools import *
import torch
import pydicom
import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import recon_ops
device = torch.device('cuda:0')

ops = recon_ops()

root_dir = '/Data'
case_list = os.listdir(root_dir + '/AAPM/')

for index, case in enumerate(case_list):

    hdproj_save_path = root_dir + '/sAAPMProj/' + case + '/clean_proj/'
    ldproj_save_path = root_dir + '/sAAPMProj/' + case + '/noisy_proj_1e5/'
    ldct_save_path = root_dir + '/sAAPMCT/' + case + '/noisy_CT_1e5/'
    make_dirs(hdproj_save_path)
    make_dirs(ldproj_save_path)
    make_dirs(ldct_save_path)

    hdct_path = root_dir + '/AAPM/' + case + '/full_1mm/'
    # ldct_path = root_dir + case + '/quarter_1mm/'

    hdct_vol = read_dicom_all(hdct_path, 20, 24)

    # plt.imshow(hdct_vol[150, :, :], cmap=plt.cm.gray, vmin=1024-160, vmax=1024+240)
    # plt.show()

    hdct_vol = hdct_vol / 1024 * 0.02
    # ldct_vol = read_dicom_all(ldct_path, 20, 24)
    # ldct_vol = ldct_vol / 1024 * 0.02

    for slice in range(np.size(hdct_vol, 0)):

        hdct_slice = hdct_vol[slice, :, :]
        # ldct_slice = ldct_vol[slice, :, :]

        with torch.no_grad():

            hdct_slice_cuda = torch.FloatTensor(hdct_slice).to(device)
            hdproj_slice_cuda = ops.forward(hdct_slice_cuda)
            # hdct_fbp_slice_cuda = ops.backprojection(ops.filter_sinogram(hdproj_slice_cuda))
            hdproj_slice = hdproj_slice_cuda.cpu().detach().numpy()

            ldproj_slice = addPossionNoisy(hdproj_slice, 1e5)
            ldproj_slice_cuda = torch.FloatTensor(ldproj_slice).to(device)

            ldct_slice_cuda = ops.backprojection(ops.filter_sinogram(ldproj_slice_cuda))
            ldct_slice = ldct_slice_cuda.cpu().detach().numpy()

            # plt.imshow(np.concatenate([ldct_slice, hdct_slice, hdct_slice-ldct_slice], axis=1) / 0.02 * 1024, cmap=plt.cm.gray, vmin=1024-160, vmax=1024+240)
            # plt.show()

        (ldct_slice / 0.02 * 1024 - 1024).astype(np.float32).tofile(ldct_save_path + str(slice+1) + '_noisy_ct' + '.raw')
        ldproj_slice.astype(np.float32).tofile(ldproj_save_path + str(slice+1) + '_noisy_proj' + '.raw')
        hdproj_slice.astype(np.float32).tofile(hdproj_save_path + str(slice+1) + '_clean_proj' + '.raw')

        # plt.imshow(hdct_fbp_slice, cmap=plt.cm.gray)
        # plt.imshow((hdproj_slice-ldproj_slice), cmap=plt.cm.gray)
        # plt.imshow(np.concatenate([hdct_fbp_slice, hdct_slice, hdct_slice-hdct_fbp_slice], axis=1) / 0.02 * 1024, cmap=plt.cm.gray, vmin=1024-160, vmax=1024+240)
        # plt.show()