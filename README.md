# CLEAR

A Pytorch-Version of CLEAR. Some details are different from the original paper. Note: clear_with_gan is not easy to train

1. Install torch-randon (https://github.com/matteo-ronchetti/torch-radon)
2. Generate simulated low-dose CT data: gen_ld_data.py
3. Generate training data: make_proj_img_list_3d.py
4. Train and Validation: train_clear_without_gan.py, train_clear_with_gan.py, train_clear_with_gan_multi_step.py
5. Test: test_clear.py

Please cite the following references:
1. TorchRadon: Fast Differentiable Routines for Computed Tomography
2. CLEAR: comprehensive learning enabled adversarial reconstruction for subtle structure enhanced low-dose CT imaging
