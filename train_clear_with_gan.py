import os
from datasets import npz_proj_img_reader_func
import scipy
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse

from models import *
from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_loader, writer, epoch, g_model, g_optm, g_lr, d_model, d_optm, d_lr):
    batch_time = AverageMeter()
    loss_recon_scalar = AverageMeter()
    loss_adv_scalar = AverageMeter()
    end = time.time()

    g_model.train()
    d_model.train()

    step = 0

    for data in train_loader:

        hdProj = data["hdproj"]
        hdProj = hdProj.cuda()

        ldProj = data["ldproj"]
        ldProj = ldProj.cuda()

        hdCT = data["hdct"]
        hdCT = hdCT.cuda()

        proj_net, _, img_net = g_model(ldProj)

        d_optm.zero_grad()

        # patches_show = extract_patches_online(hdCT, 4)

        real = d_model(extract_patches_online(hdCT, 4))
        fake = d_model(extract_patches_online(img_net, 4))

        gradient_penalty = compute_gradient_penalty(d_model, extract_patches_online(hdCT, 4), extract_patches_online(img_net, 4))
        d_loss_adv = -torch.mean(real) + torch.mean(fake) + 10 * gradient_penalty
        d_loss_adv.backward()
        d_optm.step()

        loss_adv_scalar.update(d_loss_adv.item(), hdCT.size(0))

        g_optm.zero_grad()

        if step % 1 == 0:
            
            proj_net, img_fbp, img_net = g_model(ldProj)

            fake = d_model(extract_patches_online(img_net, 4))

            g_loss_adv = -torch.mean(fake)

            loss_proj = 20 * F.l1_loss(proj_net, hdProj)
            loss_img = F.mse_loss(img_net, hdCT)
            g_loss_recon = 0.01 * loss_proj + loss_img

            g_loss = g_loss_recon + 0.01 * g_loss_adv

            g_loss.backward()
            g_optm.step()

            loss_recon_scalar.update(loss_img.item(), hdCT.size(0))

        loss_recon_scalar.update(loss_img.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    g_lr.step()
    d_lr.step()

    writer.add_scalars('recon_loss', {'train_mse_loss': loss_recon_scalar.avg}, epoch + 1)
    writer.add_scalars('adv_loss', {'loss_d_adv': loss_adv_scalar.avg}, epoch + 1)

    writer.add_scalar('learning_rate', g_lr.get_last_lr()[0], epoch + 1)

    writer.add_image('valid img/label-fbp-result img', normalization(torch.cat([hdCT[0, :, 1, :, :], img_fbp[0, :, 1, :, :], img_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('train img/label-result proj', normalization(torch.cat([hdProj[0, :, 1, :, :], proj_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(hdCT[0, :, 1, :, :] - img_net[0, :, 1, :, :])), epoch + 1)
    writer.add_image('train img/residual proj', normalization(torch.abs(hdProj[0, :, 1, :, :] - proj_net[0, :, 1, :, :])), epoch + 1)
    # writer.add_image('train img/patches', normalization(patches_show[16, :, 2, :, :]), epoch + 1)

    print('Train Epoch: {}\t train_mse_loss: {:.6f}\t'.format(epoch + 1, loss_recon_scalar.avg))


def valid(valid_loader, g_model, writer, epoch):
    batch_time = AverageMeter()
    loss_recon_scalar = AverageMeter()
    end = time.time()

    g_model.eval()

    step = 0

    for data in valid_loader:
        hdProj = data["hdproj"]
        hdProj = hdProj.cuda()

        ldProj = data["ldproj"]
        ldProj = ldProj.cuda()

        hdCT = data["hdct"]
        hdCT = hdCT.cuda()

        with torch.no_grad():

            proj_net, img_fbp, img_net = g_model(ldProj)
            loss_img = F.mse_loss(img_net, hdCT)

        loss_recon_scalar.update(loss_img.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    writer.add_scalars('recon_loss', {'valid_mse_loss': loss_recon_scalar.avg}, epoch + 1)
    writer.add_image('valid img/label-fbp-result img', normalization(torch.cat([hdCT[0, :, 1, :, :], img_fbp[0, :, 1, :, :], img_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/label-result proj', normalization(torch.cat([hdProj[0, :, 1, :, :], proj_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(hdCT[0, :, 1, :, :] - img_net[0, :, 1, :, :])), epoch + 1)
    writer.add_image('valid img/residual proj', normalization(torch.abs(hdProj[0, :, 1, :, :] - proj_net[0, :, 1, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mse_loss: {:.6f}\t'.format(epoch + 1, loss_recon_scalar.avg))


if __name__ == "__main__":

    cudnn.benchmark = True

    method = 'clear_with_gan'
    batch_size = 2

    result_path = './runs/' + method + '/logs/'
    save_dir = './runs/' + method + '/checkpoints/'

    # Get dataset
    train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./txt/train_3d_list_s5e4.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    valid_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./txt/valid_3d_list_s5e4.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    generator = GeneratorCLEAR(chl=32)
    discriminator = DiscriminatorCLEAR()
    # criterion = torch.nn.MSELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=1, gamma=0.9541)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=1, gamma=0.9541)

    if os.path.exists(save_dir) is False:

        generator = generator.cuda()
        discriminator = discriminator.cuda()

    else:
        checkpoint_latest_g = torch.load(find_lastest_file(save_dir + 'G/'))
        generator = load_model(generator, checkpoint_latest_g)
        generator = generator.cuda()
        optimizer_g.load_state_dict(checkpoint_latest_g['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint_latest_g['lr_scheduler'])

        checkpoint_latest_d = torch.load(find_lastest_file(save_dir + 'D/'))
        discriminator = load_model(discriminator, checkpoint_latest_d)
        discriminator = discriminator.cuda()
        optimizer_d.load_state_dict(checkpoint_latest_d['optimizer_state_dict'])
        scheduler_d.load_state_dict(checkpoint_latest_d['lr_scheduler'])

        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*" * 20 + "Start Train" + "*" * 20)

    for epoch in range(0, 100):
        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, writer, epoch, generator, optimizer_g, scheduler_g, discriminator, optimizer_d, scheduler_d)
        valid(valid_loader, generator, writer, epoch)

        save_model(generator, optimizer_g, scheduler_g, epoch + 1, save_dir + 'G/')
        save_model(discriminator, optimizer_d, scheduler_d, epoch + 1, save_dir + 'D/')
