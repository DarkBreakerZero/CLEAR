# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.autograd as autograd

# input (Tensor)
# pad (tuple)
# mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
# value – fill value for 'constant' padding. Default: 0

class ConvBnRelu3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=False, is_relu=True):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm3d(out_chl, eps=1e-4)
        if is_relu is True:
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class EncoderResBlock3D(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(EncoderResBlock3D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        res_out = x + conv_out

        return res_out


class DeconderResBlock3D(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(DeconderResBlock3D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl * 2, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, cat_x, upsample_x):
        conv_out = self.encode(cat_x)
        res_out = upsample_x + conv_out

        return res_out


class CLEAR_UNetL3(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(CLEAR_UNetL3, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.c1 = ConvBnRelu3d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.res1 = EncoderResBlock3D(model_chl, model_chl)
        self.d1 = ConvBnRelu3d(model_chl, model_chl * 2, kernel_size=kernel_size, dilation=1, stride=(1, 2, 2),
                               groups=1)

        self.res2 = EncoderResBlock3D(model_chl * 2, model_chl * 2)
        self.d2 = ConvBnRelu3d(model_chl * 2, model_chl * 4, kernel_size=kernel_size, dilation=1, stride=(1, 2, 2),
                               groups=1)

        self.res3 = EncoderResBlock3D(model_chl * 4, model_chl * 4)
        self.d3 = ConvBnRelu3d(model_chl * 4, model_chl * 8, kernel_size=kernel_size, dilation=1, stride=(1, 2, 2),
                               groups=1)

        self.res4 = EncoderResBlock3D(model_chl * 8, model_chl * 8)

        self.u3 = nn.ConvTranspose3d(model_chl * 8, model_chl * 4, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 1, 1), stride=(1, 2, 2))
        self.ures3 = DeconderResBlock3D(model_chl * 4, model_chl * 4)

        self.u2 = nn.ConvTranspose3d(model_chl * 4, model_chl * 2, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 1, 1), stride=(1, 2, 2))
        self.ures2 = DeconderResBlock3D(model_chl * 2, model_chl * 2)

        self.u1 = nn.ConvTranspose3d(model_chl * 2, model_chl * 1, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 1, 1), stride=(1, 2, 2))
        self.ures1 = DeconderResBlock3D(model_chl * 1, model_chl * 1)

        self.out = ConvBnRelu3d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):
        c1 = self.c1(x)
        res1 = self.res1(c1)
        d1 = self.d1(res1)
        # print(c1.size(), res1.size(), d1.size())

        res2 = self.res2(d1)
        d2 = self.d2(res2)
        # print(res2.size(), d2.size())

        res3 = self.res3(d2)
        d3 = self.d3(res3)

        res4 = self.res4(d3)

        u3 = self.u3(res4)
        cat3 = torch.cat([u3, res3], 1)
        ures3 = self.ures3(cat3, u3)

        u2 = self.u2(ures3)
        cat2 = torch.cat([u2, res2], 1)
        ures2 = self.ures2(cat2, u2)

        u1 = self.u1(ures2)
        cat1 = torch.cat([u1, res1], 1)
        ures1 = self.ures1(cat1, u1)

        out = F.leaky_relu(self.out(ures1) + x)

        return out
        
from utils import recon_ops

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

recon_ops_example = recon_ops()

class GeneratorCLEAR(nn.Module):

    def __init__(self, chl=32):
        super(GeneratorCLEAR, self).__init__()
        self.chl = chl

        self.net = nn.ModuleList()
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))

    def forward(self, proj):
        # img_fbp_0 = recon_ops_example.backprojection(recon_ops_example.filter_sinogram(proj))
        proj_net = self.net[0](proj)
        # print(proj_net.size())
        img_fbp = recon_ops_example.backprojection(recon_ops_example.filter_sinogram(proj_net)) * 1024
        # print(img_fbp.size())
        img_net = self.net[1](img_fbp)
        # print(img_net.size())
        proj_re = recon_ops_example.forward(img_net / 1024)
        # print(proj_re.size())

        return proj_net, img_fbp, img_net, proj_re
        # return proj_net, img_fbp, img_net, proj_re, img_fbp_0

class DiscriminatorCLEAR(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(DiscriminatorCLEAR, self).__init__()
        self.ConvLayers = nn.Sequential(
            ConvBnRelu3d(in_chl, model_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl, model_chl, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1, is_bn=True),

            ConvBnRelu3d(model_chl * 1, model_chl * 2, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl * 2, model_chl * 2, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1,
                         is_bn=True),

            ConvBnRelu3d(model_chl * 2, model_chl * 4, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl * 4, model_chl * 4, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1,
                         is_bn=True),

            ConvBnRelu3d(model_chl * 4, model_chl * 8, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl * 8, model_chl * 8, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1,
                         is_bn=True)
        )
        self.FCLayer = nn.Sequential(
            nn.Linear(model_chl * 8, out_chl)
        )

    def forward(self, x):
        out = self.ConvLayers(x)
        # print(out.size())
        out = torch.reshape(F.adaptive_avg_pool3d(out, (1, 1, 1)), [out.shape[0], out.shape[1]])
        # print(out.size())
        out = self.FCLayer(out)
        # print(out.size())

        return out

def compute_gradient_penalty(D, real_samples, fake_samples):
    # print(real_samples.size())
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.ndim == 5:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    else:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
