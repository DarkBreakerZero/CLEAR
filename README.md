# CLEAR

A Pytorch-Version for CLEAR. Some details are different from the original paper. Note: CLEAR_with_GAN is not easy to train

Install torch-randon (https://github.com/matteo-ronchetti/torch-radon)

Generate simulated low-dose CT data: gen_ld_data.py

Generate training data: make_proj_img_list_3d.py

Train and Validation: train_clear_without_gan.py, train_clear_with_gan.py, train_clear_with_gan_multi_step.py

Test: test_clear.py

Please cite the following references:

TorchRadon: Fast Differentiable Routines for Computed Tomography
CLEAR: comprehensive learning enabled adversarial reconstruction for subtle structure enhanced low-dose CT imaging
