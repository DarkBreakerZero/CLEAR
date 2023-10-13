import os
import matplotlib.pyplot as plt
import numpy as np
from pytools import *
# import pylab
import time
# from scipy.ndimage import zoom
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
train_cases = ['L096', 'L143', 'L333', 'L291', 'L286', 'L310']
valid_cases = ['L506', 'L109']
test_cases = ['L067', 'L192']

root_dir = '/Data'

txt_save_dir = './txt/'
make_dirs(txt_save_dir)

train_npz_save_dir = root_dir + '/NPZ/TrainProjImgs5e4/'

make_dirs(train_npz_save_dir)

cnt = 0

for case in train_cases:

    startTime = time.time()

    hdct_path = root_dir + '/AAPM/' + case + '/full_1mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol = hdct_vol * 0.02 / 1024

    ldct_vol = read_raw_data_all(root_dir + '/sAAPMCT/' + case + '/noisy_CT_5e4/', w=512, h=512, start_index=0, end_index=-13)
    ldct_vol = (ldct_vol + 1024) / 1024 * 0.02
    hdproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/clean_proj/', w=576, h=736, start_index=0, end_index=-15)
    ldproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/noisy_proj_5e4/', w=576, h=736, start_index=0, end_index=-15)

    # plt.imshow(ldproj_vol[150, :, :], cmap=plt.cm.gray)
    # plt.show()

    for slice in range(0, np.size(hdct_vol, 0)-3, 1):

        hdct_slices = hdct_vol[slice:slice+3, :, :]
        ldct_slices = ldct_vol[slice:slice+3, :, :]
        hdproj_slices = hdproj_vol[slice:slice+3, :, :]
        ldproj_slices = ldproj_vol[slice:slice+3, :, :]

        # print(np.shape(hdct_slices))
        # print(np.shape(hdproj_slices))
        # print(np.shape(ldproj_slices))

        np.savez(train_npz_save_dir + case + '_slice' + str(slice), ldproj=ldproj_slices, hdproj=hdproj_slices, hdct=hdct_slices, ldct=ldct_slices)
        with open(txt_save_dir + 'train_3d_list_s5e4.txt', 'a') as f:
            f.write(train_npz_save_dir + case + '_slice' + str(slice) + '.npz\n')
        f.close()

        cnt += 1

    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds.'.format(case, cnt, int(endTime-startTime)))

valid_npz_save_dir = root_dir + '/NPZ/ValidProjImgs5e4/'
make_dirs(valid_npz_save_dir)

cnt = 0

for case in valid_cases:

    startTime = time.time()

    hdct_path = root_dir + '/AAPM/' + case + '/full_1mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol = hdct_vol * 0.02 / 1024

    ldct_vol = read_raw_data_all(root_dir + '/sAAPMCT/' + case + '/noisy_CT_5e4/', w=512, h=512, start_index=0, end_index=-13)
    ldct_vol = (ldct_vol + 1024) / 1024 * 0.02
    hdproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/clean_proj/', w=576, h=736, start_index=0, end_index=-15)
    ldproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/noisy_proj_5e4/', w=576, h=736, start_index=0, end_index=-15)

    for slice in range(0, np.size(hdct_vol, 0)-3, 3):

        hdct_slices = hdct_vol[slice:slice+3, :, :]
        ldct_slices = ldct_vol[slice:slice+3, :, :]
        hdproj_slices = hdproj_vol[slice:slice+3, :, :]
        ldproj_slices = ldproj_vol[slice:slice+3, :, :]

        np.savez(valid_npz_save_dir + case + '_slice' + str(slice), ldproj=ldproj_slices, hdproj=hdproj_slices, hdct=hdct_slices, ldct=ldct_slices)
        with open(txt_save_dir + 'valid_3d_list_s5e4.txt', 'a') as f:
            f.write(valid_npz_save_dir + case + '_slice' + str(slice) + '.npz\n')
        f.close()

        cnt += 1

    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds.'.format(case, cnt, int(endTime-startTime)))