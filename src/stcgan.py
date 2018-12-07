import os
import time
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


import utils
from dataset import ShadowRemovalDataset, get_composed_augmentation
from fileio import writePickle, readPickle
from logHandler import get_logger

class STCGAN():
    def __init__(self, args, path):
        self.logger = get_logger(__name__)
        self.mdl_name = 'STCGAN'
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.mdl_dir = path.mdl_dir
        self.train_hist = {
            'G1_loss': [],
            'G2_loss': [],
            'D1_loss': [],
            'D2_loss': [],
        }
        # data_loader
        # split data
        train_size, test_size = len(os.listdir(path.train_shadow_dir)), len(os.listdir(path.test_shadow_dir))
        train_img_list, test_img_list = list(range(train_size)), list(range(test_size))
        
        # TODO: add augmentation here
        training_augmentation = get_composed_augmentation()
        if args.valid_ratio:
            split_size = int((1 - args.valid_ratio) * total_size)
            train_img_list, valid_img_list = train_img_list[:split_size], train_img_list[split_size:]
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_augmentation)
            valid_dataset = ShadowRemovalDataset(path, 'validation', valid_img_list)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8) 
            self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        else:
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_augmentation)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = ShadowRemovalDataset(path, 'testing', test_img_list)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        # model
        self.G1 = STCGAN_G1()
        self.G2 = STCGAN_G2()
        self.D1 = STCGAN_D1()
        self.D2 = STCGAN_D2()

        self.G_opt = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_opt = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G1.cuda()
            self.G2.cuda()
            self.D1.cuda()
            self.D2.cuda()
            self.l1_loss = nn.L1Loss().cuda()
            self.adversial_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.l1_loss = nn.L1Loss()
            self.adversial_loss = nn.CrossEntropyLoss()
        
        self.logger.info('-' * 10 + ' Networks Architecture ' + '-' * 10)
        utils.print_netowrk(self.G1)
        utils.print_netowrk(self.G2)
        utils.print_netowrk(self.D1)
        utils.print_netowrk(self.D2)
        self.logger.info('-' * 43)


    def train(self):
        self.y_real, self.y_fake = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real, self.y_fake = self.y_real.cuda(), self.y_fake.cuda()
            
        self.D1.train()
        self.D2.train()
        self.logger.info('Start training...')
        start_time = time.time()
        
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            self.G1.train()
            self.G2.train()
        
            for i, (shadow_img, shadow_free_img, mask_img) in enumerate(self.train_loader):


            self.logger.info('[ Training ] Epoch: {:3d} Total_loss: {:.3f} G1_loss: {:.3f} G2_loss: {:.3f} D1_loss: {:.3f} D2_loss: {:.3f}'.format(
                epoch, G1_loss, G2_loss, D1_loss, D2_loss
            ))
            with torch.no_grad():
                self.G1.eval()
                self.G2.eval()
                

    def save(self, epoch):
        torch.save(self.G1.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_G1.ckpt'))
        torch.save(self.G2.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_G2.ckpt'))
        torch.save(self.D1.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_D1.ckpt'))
        torch.save(self.D2.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_D2.ckpt'))
        writePickle(self.train_hist, os.path.join(self.mdl_dir, self.mdl_name + '_history.pkl'))
    def load(self):
        self.G1.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_G1.ckpt')))
        self.G2.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_G2.ckpt')))
        self.D1.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_D1.ckpt')))
        self.D2.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_D2.ckpt')))
        self.train_hist = readPickle(os.path.join(self.mdl_dir, self.mdl_name + '_history.pkl'))

class STCGAN_G(nn.Module):
    def __init__(self, n_in, n_out):
        super(STCGAN_G, self).__init__()
        self.inconv = inconv(n_in, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512, n_block=3)
        self.down5 = down(512, 512)
        self.up1 = up(512, 512)
        self.up2 = up(1024, 512, n_block=3)
        self.up3 = up(1024, 256)
        self.up4 = up(512, 128)
        self.up5 = up(256, 64)
        self.outconv = outconv(128, n_out)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1()
        
# class STCGAN_G():

class STCGAN_D(nn.Module):
    def __init__(self, n_in, n_out):
    
    def forward():

# class STCGAN_D():





# parts #


class inconv(nn.Module):
    def __init__(self, n_in, n_out):
        super(inconv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, 3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__():
        super(outconv, self).__init__()
        self.conv = nn.
    def forward():

class down(nn.Module):
    def __init__(self, n_in, n_out, n_block=1, act=nn.LeakyReLU, bn=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv2d(n_in, n_out, n_block, act, bn)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__():
    def forward():        

class conv2d(nn.Module):
    def __init__(self, n_in, n_out, n_block=1, act=nn.LeakyReLU, bn=True):
        super(conv2d, self).__init__()
        block = []
        for _ in range(n_block):
            block.append(nn.Conv2d(n_in, n_out, 3, padding=1))
            block.append(act(inplace=True)
            if bn:
                block.append(nn.BatchNorm2d(n_out))
        
        self.block = nn.Sequential(*block)

    def forward(self, x)
        x = self.block(x)
        return x

class deconv2d(nn.Module):
    def __init__(self, n_in, n_out, act=nn.ReLU, bn=True):
        super(deconv2d, self).__init__()
        
        if bn:
            self.devonc = nn.Sequential(
                nn.Conv2d
            )
