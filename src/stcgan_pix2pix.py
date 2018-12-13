import os
import cv2
import time
import glob
import random
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


import utils

from dataset import ShadowRemovalDataset, get_composed_transform
from eval import MetricType, eval_func, eval_dir_func
from fileio import writePickle, readPickle
from logHandler import get_logger

from resource_pix2pix import networks

class STCGAN():
    def __init__(self, args, path):
        self.logger = get_logger(__name__)
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.batch_size_test = args.batch_size_test
        self.valid_ratio = args.valid_ratio
        self.valid_step = args.valid_step
        self.model_step = args.model_step
        self.model_name = args.model_name
        
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3


        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        
        # self.gpu_mode = args.gpu_mode
        # self.gpu_id = args.gpu_id
        self.path = path
        self.mdl_dir = path.mdl_dir
        self.train_hist = {
            'loss': [],
            'G_loss': [],
            'D_loss': [],
            'G1_loss': [],
            'G2_loss': [],
            'D1_loss': [],
            'D2_loss': [],
        }
        self.valid_hist = {
            'loss': [],
            'mask_loss': [],
            'shadow_free_loss': []
        }
        self.min_loss = [np.inf] * 5
        self.device = torch.device('cuda:{}'.format(args.gpu_id))
        # data_loader
        # split data
        train_size, test_size = len(os.listdir(path.train_shadow_dir)), len(os.listdir(path.test_shadow_dir))
        train_img_list, test_img_list = list(range(train_size)), list(range(test_size))
        
        # add augmentation here
        training_transforms = get_composed_transform()
        if self.valid_ratio:
            split_size = int((1 - args.valid_ratio) * train_size)
            train_img_list, valid_img_list = train_img_list[:split_size], train_img_list[split_size:]
            random.shuffle(valid_img_list)
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_transforms)
            valid_dataset = ShadowRemovalDataset(path, 'validation', valid_img_list)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8) 
            self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size_test, shuffle=False, num_workers=8)
            self.logger.info('Training size: {} Validation size: {}'.format(split_size, train_size - split_size))
        else:
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_transforms)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = ShadowRemovalDataset(path, 'testing', test_img_list)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size_test, shuffle=False, num_workers=8)



        # model
        self.G1 = networks.define_G(input_nc=3, output_nc=1, ngf=64, netG='unet_256', gpu_ids=[args.gpu_id])
        self.G2 = networks.define_G(input_nc=4, output_nc=3, ngf=64, netG='unet_256', gpu_ids=[args.gpu_id])
        self.D1 = networks.define_D(input_nc=3+1, ndf=64, netD='pixel', use_sigmoid=True, gpu_ids=[args.gpu_id])
        self.D2 = networks.define_D(input_nc=3+3+1, ndf=64, netD='pixel', use_sigmoid=True, gpu_ids=[args.gpu_id])

        self.G_opt = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_opt = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.l1_loss = nn.L1Loss()
        # self.mse_loss = nn.MSELoss()  # for validation
        # self.bce_loss = nn.BCELoss()
        self.gan_loss = networks.GANLoss(use_lsgan=False).to(self.device)
        
        # self.logger.info('-' * 10 + ' Networks Architecture ' + '-' * 10)
        # utils.print_netowrk(self.G1)
        # utils.print_netowrk(self.G2)
        # utils.print_netowrk(self.D1)
        # utils.print_netowrk(self.D2)
        # self.logger.info('-' * 43)


    def train(self):
          
        self.D1.train()
        self.D2.train()
        self.logger.info('Start training...')
        start_time = time.time()
        
        total_steps = 0
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            self.G1.train()
            self.G2.train()
            # self.logger.info('epoch {}'.format(epoch))
            for i, (self.shadow_img, self.shadow_free_img, self.mask_img) in enumerate(self.train_loader):
                # self.logger.info('Iteration: {}'.format(i))
                # A_real(shadow), B_real(shadow_free), C_real(mask)
                self.set_input()
                # self.A_real = shadow_img.to(self.device)
                # self.B_real = shadow_free_img.to(self.device)
                # self.C_real = mask_img.to(self.device)

                self.optimize_parameter()
                self.logger.info('[ Training ] Epoch: {:3d} iteration: {:3d} Loss: {:.3f} [G: {:.3f}] [D: {:.3f}] [G1: {:.3f}] [G2: {:.3f}] [D1: {:.3f}] [D2: {:.3f}]'.format(
                    epoch, i+1, self.G_loss.item() + self.D_loss.item(), self.G_loss.item(), self.D_loss.item(), self.G1_loss.item(), self.G2_loss.item(), self.D1_loss.item(), self.D2_loss.item()
                ))
                total_steps += 1
                if self.valid_ratio and total_steps % self.valid_step == 0:
                    self.valid(total_steps)
                    if self.v_loss < self.min_loss[-1]:
                        self.logger.info('[ SAVE LOSS ] current: {}, list {}'.format(self.v_loss, self.min_loss))
                        self.min_loss.append(self.v_loss)
                        self.min_loss = sorted(self.min_loss)[:-1]
                        # self.min_loss = self.v_loss
                        self.save('best_{:06d}'.format(total_steps))

            if (epoch+1) % 10 == 0 or (epoch+1) == self.epoch:
                self.save('latest_{:06d}'.format(total_steps))

        
    def valid(self, steps):
        with torch.no_grad():
            self.G1.eval()
            self.G2.eval()
            v1_loss = []
            v2_loss = []
            for i, (valid_shadow_img, valid_shadow_free_img, valid_mask_img) in enumerate(self.valid_loader):
                vA_real = valid_shadow_img.to(self.device)
                vB_real = valid_shadow_free_img.to(self.device)
                vC_real = valid_mask_img.to(self.device)

                vC_fake = self.G1(vA_real)
                vB_fake = self.G2(torch.cat((vA_real, vC_fake), 1))
                
                vG1_loss = self.l1_loss(vC_fake, vC_real).item()
                vG2_loss = self.l1_loss(vB_fake, vB_real).item()
                v1_loss.append(vG1_loss)
                v2_loss.append(vG2_loss)
                if i == 0:
                    # save image
                    for j in range(self.batch_size_test):
                        # gt shadow
                        valid_name = 'step_{}_valid_{}.png'.format(steps, i * self.batch_size_test + j)
                        cv2.imwrite(
                            os.path.join(self.path.valid_gt_shadow_dir, valid_name), 
                            np.transpose((vA_real.data.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                        )
                        # gt non-shadow 
                        # cv2.imwrite(
                        #     os.path.join(self.path.valid_gt_shadow_free_dir, valid_name), 
                        #     np.transpose((vB_real.data.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                        # )
                        # gt mask
                        # cv2.imwrite(
                        #     os.path.join(self.path.valid_gt_mask_dir, valid_name), 
                        #     np.transpose((vC_real.data.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                        # )
                        # pred non-shadow
                        cv2.imwrite(
                            os.path.join(self.path.valid_shadow_free_dir, valid_name), 
                            np.transpose((vB_fake.data.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                        )
                        # pred mask
                        cv2.imwrite(
                            os.path.join(self.path.valid_mask_dir, valid_name), 
                            np.transpose((vC_fake.data.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                        )
            v1_loss = np.mean(v1_loss)
            v2_loss = np.mean(v2_loss)
            self.v_loss = v1_loss + v2_loss
            self.valid_hist['mask_loss'].append(v1_loss)
            self.valid_hist['shadow_free_loss'].append(v2_loss)
            self.valid_hist['loss'].append(self.v_loss)
            self.logger.info('[ Validation ] Iteration: {:3d}, loss: {:.3f} [mask: {:.3f}] [non-shadow: {:.3f}]'.format(steps, self.v_loss, v1_loss, v2_loss))



    def test(self):
        with torch.no_grad():
            self.G1.eval()
            self.G2.eval()
            for i, (self.shadow_img, self.shadow_free_img, self.mask_img) in enumerate(self.test_loader):
                self.set_input()
                self.forward()
                for j in range(self.batch_size_test):
                    result_name = '{:06d}.png'.format(i * self.batch_size_test + j)
                    # shadow
                    cv2.imwrite(
                        os.path.join(self.path.result_shadow_dir, result_name),
                        np.transpose((self.A_real.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                    )
                    # shadow free
                    cv2.imwrite(
                        os.path.join(self.path.result_shadow_free_dir, result_name),
                        np.transpose((self.B_fake.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                    )
                    # mask
                    cv2.imwrite(
                        os.path.join(self.path.result_mask_dir, result_name),
                        np.transpose((self.C_fake.cpu().numpy()[j, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                    )
            eval_dir_func(self.path.test_shadow_free_dir, self.path.result_shadow_free_dir, self.path.test_mask_dir, MetricType.MSE | MetricType.RMSE)

        pass
    def optimize_parameter(self):
        self.forward()
        # TODO: add require grad
        self.D_opt.zero_grad()
        self.backward_D()
        self.D_opt.step()

        # TODO: add require grad
        self.G_opt.zero_grad()
        self.backward_G()
        self.G_opt.step()
        self.train_hist['loss'].append(self.G_loss.item() + self.D_loss.item())
        
    def set_input(self):
        self.A_real = self.shadow_img.to(self.device)
        self.B_real = self.shadow_free_img.to(self.device)
        self.C_real = self.mask_img.to(self.device)

    def forward(self):
        self.C_fake = self.G1(self.A_real)

        self.AC_real = torch.cat((self.A_real, self.C_real), 1)
        self.AC_fake = torch.cat((self.A_real, self.C_fake), 1)

        self.B_fake = self.G2(self.AC_fake)

        self.ABC_real = torch.cat((self.AC_real, self.B_real), 1)
        self.ABC_fake = torch.cat((self.AC_fake, self.B_fake), 1)

    def backward_D(self):
        D1_real = self.D1(self.AC_real)
        D1_fake = self.D1(self.AC_fake.detach())

        D1_real_loss = self.gan_loss(D1_real, True) * 0.5
        D1_fake_loss = self.gan_loss(D1_fake, False) * 0.5

        self.D1_loss = (D1_real_loss + D1_fake_loss) * self.lambda2

        D2_real = self.D2(self.ABC_real)
        D2_fake = self.D2(self.ABC_fake.detach())

        D2_real_loss = self.gan_loss(D2_real, True) * 0.5
        D2_fake_loss = self.gan_loss(D2_fake, False) * 0.5  

        self.D2_loss = (D2_real_loss + D2_fake_loss) * self.lambda3
        self.D_loss = self.D1_loss + self.D2_loss

        self.train_hist['D_loss'].append(self.D_loss.item())
        self.train_hist['D1_loss'].append(self.D1_loss.item())
        self.train_hist['D2_loss'].append(self.D2_loss.item())

        self.D_loss.backward()

    def backward_G(self):
        self.G1_loss = self.l1_loss(self.C_fake, self.C_real)
        self.G2_loss = self.l1_loss(self.B_fake, self.B_real) * self.lambda1
        
        self.G_loss = self.G1_loss + self.G2_loss
        self.train_hist['G_loss'].append(self.G_loss.item())
        self.train_hist['G1_loss'].append(self.G1_loss.item())
        self.train_hist['G2_loss'].append(self.G2_loss.item())
        
        self.G_loss.backward()
    
    def save(self, name):
        self.logger.info('Saving model: {}'.format(name))
        torch.save(self.G1.state_dict(), os.path.join(self.mdl_dir, name + '_G1.ckpt'))
        torch.save(self.G2.state_dict(), os.path.join(self.mdl_dir, name + '_G2.ckpt'))
        torch.save(self.D1.state_dict(), os.path.join(self.mdl_dir, name + '_D1.ckpt'))
        torch.save(self.D2.state_dict(), os.path.join(self.mdl_dir, name + '_D2.ckpt'))
        writePickle(self.train_hist, os.path.join(self.mdl_dir, self.__class__.__name__ + '_history.pkl'))
    
    def load(self):
        # name, best or latest
        
        if self.model_name == 'latest':
            # load latest
            mdl = sorted(glob.glob(os.path.join(self.mdl_dir, 'latest*')))[-1].split('/')[-1][:-8]
            
        elif self.model_name == 'best':
            # load best
            mdl = sorted(glob.glob(os.path.join(self.mdl_dir, 'best*')))[-1].split('/')[-1][:-8]
            
        else:
            # load by name
            mdl = self.model_name
        
        self.logger.info('Loading model: {}'.format(mdl))
        self.G1.load_state_dict(torch.load(os.path.join(self.mdl_dir, mdl + '_G1.ckpt')))
        self.G2.load_state_dict(torch.load(os.path.join(self.mdl_dir, mdl + '_G2.ckpt')))
        self.D1.load_state_dict(torch.load(os.path.join(self.mdl_dir, mdl + '_D1.ckpt')))
        self.D2.load_state_dict(torch.load(os.path.join(self.mdl_dir, mdl + '_D2.ckpt')))
        self.train_hist = readPickle(os.path.join(self.mdl_dir, self.__class__.__name__ + '_history.pkl'))
