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

from models import GeneratorCLEAR
from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_loader, model, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    loss_img = AverageMeter()
    loss_proj = AverageMeter()
    model.train()
    end = time.time()

    step = 0

    for data in train_loader:

        hdProj = data["hdproj"]
        hdProj = hdProj.cuda()

        ldProj = data["ldproj"]
        ldProj = ldProj.cuda()

        hdCT = data["hdct"]
        hdCT = hdCT.cuda()

        proj_net, img_net = model(ldProj)

        loss_proj = 20 * F.l1_loss(proj_net, hdProj)
        loss_img = F.mse_loss(img_net, hdCT)

        loss = 0.01 * loss_proj + loss_img

        loss_img.update(loss_img.item(), hdCT.size(0))
        loss_proj.update(loss_proj.item(), hdProj.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('train_loss', {'loss_img': loss_img.avg}, epoch + 1)
    writer.add_scalars('train_loss', {'loss_proj': loss_proj.avg}, epoch + 1)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)

    writer.add_image('train img/label-result img', normalization(torch.cat([hdCT[0, :, 1, :, :], img_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('train img/label-result proj', normalization(torch.cat([hdProj[0, :, 1, :, :], proj_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(hdCT[0, :, 1, :, :] - img_net[0, :, 1, :, :])), epoch + 1)
    writer.add_image('train img/residual proj', normalization(torch.abs(hdProj[0, :, 1, :, :] - proj_net[0, :, 1, :, :])), epoch + 1)

    scheduler.step()

    print('Train Epoch: {}\t train_loss: {:.6f}\t'.format(epoch + 1, loss_img.avg))

def valid(valid_loader, model, writer, epoch):

    batch_time = AverageMeter()
    loss_img = AverageMeter()
    loss_proj = AverageMeter()
    model.eval()
    end = time.time()

    step = 0

    for data in valid_loader:

        hdProj = data["hdproj"]
        hdProj = hdProj.cuda()

        ldProj = data["ldproj"]
        ldProj = ldProj.cuda()

        hdCT = data["hdct"]
        hdCT = hdCT.cuda()


        with torch.no_grad():

            proj_net, img_net = model(ldProj)

            loss_proj = 20 * F.l1_loss(proj_net, hdProj)
            loss_img = F.mse_loss(img_net, hdCT)

        loss_img.update(loss_img.item(), hdCT.size(0))
        loss_proj.update(loss_proj.item(), hdProj.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    writer.add_scalars('valid_loss', {'loss_img': loss_img.avg}, epoch + 1)
    writer.add_scalars('valid_loss', {'loss_proj': loss_proj.avg}, epoch + 1)

    writer.add_image('valid img/label-result img', normalization(torch.cat([hdCT[0, :, 1, :, :], img_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/label-result proj', normalization(torch.cat([hdProj[0, :, 1, :, :], proj_net[0, :, 1, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(hdCT[0, :, 1, :, :] - img_net[0, :, 1, :, :])), epoch + 1)
    writer.add_image('valid img/residual proj', normalization(torch.abs(hdProj[0, :, 1, :, :] - proj_net[0, :, 1, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_loss: {:.6f}\t'.format(epoch + 1, loss_img.avg))

if __name__ == "__main__":

    cudnn.benchmark = True

    method = 'clear_without_gan'
    batch_size = 4

    result_path = './runs/' + method + '/logs/'
    save_dir = './runs/' + method + '/checkpoints/'

    # Get dataset
    train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./txt/train_3d_list_s5e4.txt')
    # train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./txt/valid_3d_list_s5e4.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    valid_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./txt/valid_3d_list_s5e4.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    model = GeneratorCLEAR(chl=32)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9541)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 12, 16, 20, 100], gamma=0.4)

    if os.path.exists(save_dir) is False:

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, writer, epoch)

        save_model(model, optimizer, scheduler, epoch + 1, save_dir)
