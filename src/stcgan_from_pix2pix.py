import os
import cv2
import time
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


import utils

from dataset import ShadowRemovalDataset, get_composed_augmentation
from eval import MetricType, eval_func
from fileio import writePickle, readPickle
from logHandler import get_logger

from resource_pix2pix import networks

class STCGAN():
    def __init__(self, args, path):
        self.logger = get_logger(__name__)
        self.mdl_name = 'STCGAN'
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.valid_ratio = args.valid_ratio

        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3


        self.gpu_mode = args.gpu_mode
        # self.gpu_id = args.gpu_id
        self.path = path
        self.mdl_dir = path.mdl_dir
        self.train_hist = {
            'G_loss': [],
            'D_loss': [],
            'G1_loss': [],
            'G2_loss': [],
            'D1_loss': [],
            'D2_loss': [],

        }
        self.device = torch.device('cuda:{}'.format(args.gpu_id)) if args.gpu_id else torch.device('cpu')
        # data_loader
        # split data
        train_size, test_size = len(os.listdir(path.train_shadow_dir)), len(os.listdir(path.test_shadow_dir))
        train_img_list, test_img_list = list(range(train_size)), list(range(test_size))
        

        # TODO: add augmentation here
        training_augmentation = get_composed_augmentation()
        if self.valid_ratio:
            split_size = int((1 - args.valid_ratio) * train_size)
            train_img_list, valid_img_list = train_img_list[:split_size], train_img_list[split_size:]
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_augmentation)
            valid_dataset = ShadowRemovalDataset(path, 'validation', valid_img_list)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8) 
            self.valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=8)
            self.logger.info('Training size: {} Validation size: {}'.format(split_size, train_size - split_size))
        else:
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_augmentation)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = ShadowRemovalDataset(path, 'testing', test_img_list)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)



        # model
        self.G1 = networks.define_G(input_nc=3, output_nc=1, ngf=64, netG='unet_256', gpu_ids=[args.gpu_id])
        self.G2 = networks.define_G(input_nc=4, output_nc=3, ngf=64, netG='unet_256', gpu_ids=[args.gpu_id])
        self.D1 = networks.define_D(input_nc=3+1, ndf=64, netD='pixel', use_sigmoid=True, gpu_ids=[args.gpu_id])
        self.D2 = networks.define_D(input_nc=3+3+1, ndf=64, netD='pixel', use_sigmoid=True, gpu_ids=[args.gpu_id])

        self.G_opt = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_opt = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()  # for validation
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
        
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            self.G1.train()
            self.G2.train()
            # self.logger.info('epoch {}'.format(epoch))
            for i, (shadow_img, shadow_free_img, mask_img) in enumerate(self.train_loader):
                # self.logger.info('Iteration: {}'.format(i))
                
                shadow_img = shadow_img.to(self.device)
                shadow_free_img = shadow_free_img.to(self.device)
                mask_img = mask_img.to(self.device)

                # update D network
                # self.logger.info('Start update D')
                self.D_opt.zero_grad()

                # D1
                mask_fake = self.G1(shadow_img)

                shadow_mask_real = torch.cat((shadow_img, mask_img), 1)
                shadow_mask_fake = torch.cat((shadow_img, mask_fake), 1)

                D1_real = self.D1(shadow_mask_real)
                D1_fake = self.D1(shadow_mask_fake.detach())

                D1_real_loss = self.gan_loss(D1_real, True) * 0.5
                D1_fake_loss = self.gan_loss(D1_fake, False) * 0.5

                D1_loss = (D1_real_loss + D1_fake_loss) * self.lambda2
                # D2
                shadow_mask_fake = torch.cat((shadow_img, mask_fake), 1)
                shadow_free_fake = self.G2(shadow_mask_fake)

                shadow_mask_shadow_free_real = torch.cat((shadow_mask_real, shadow_free_img), 1)
                shadow_mask_shadow_free_fake = torch.cat((shadow_mask_fake, shadow_free_fake), 1)

                D2_real = self.D2(shadow_mask_shadow_free_real)
                D2_fake = self.D2(shadow_mask_shadow_free_fake.detach())

                D2_real_loss = self.gan_loss(D2_real, True) * 0.5
                D2_fake_loss = self.gan_loss(D2_fake, False) * 0.5  

                D2_loss = (D2_real_loss + D2_fake_loss) * self.lambda3

                D_loss = D1_loss + D2_loss

                self.train_hist['D_loss'].append(D_loss.item())
                self.train_hist['D1_loss'].append(D1_loss.item())
                self.train_hist['D2_loss'].append(D2_loss.item())

                D_loss.backward()
                self.D_opt.step()                

                # update G network ###########################################
                # self.logger.info('Start update G')
                self.G_opt.zero_grad()

                # G1
                mask_fake = self.G1(shadow_img)
                shadow_mask_fake = torch.cat((shadow_img, mask_fake), 1)
                D1_fake = self.D1(shadow_mask_fake)
                
                G1_loss = self.l1_loss(mask_fake, mask_img)
                # G1_bce_loss = self.bce_loss(mask_fake, mask_img)
                # G1_gan_loss = self.gan_loss(D1_fake, True) * self.lambda2
                # G1_loss = G1_bce_loss + G1_gan_loss
                # G2
                shadow_free_fake = self.G2(shadow_mask_fake)
                shadow_mask_shadow_free_fake = torch.cat((shadow_mask_fake, shadow_free_fake), 1)
                show_img = np.transpose((shadow_free_fake.data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8), [1, 2, 0])
                # self.logger.info(show_img.shape)
                # cv2.imshow('train.png', show_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                D2_fake = self.D2(shadow_mask_shadow_free_fake)
                
                G2_loss = self.l1_loss(shadow_free_fake, shadow_free_img) * self.lambda1
                # G2_gan_loss = self.gan_loss(D2_fake, True) * self.lambda3

                # G2_loss = G2_l1_loss + G2_gan_loss

                G_loss = G1_loss + G2_loss
                self.train_hist['G_loss'].append(G_loss.item())
                self.train_hist['G1_loss'].append(G1_loss.item())
                self.train_hist['G2_loss'].append(G2_loss.item())
                
                G_loss.backward()
                self.G_opt.step()
                #########################################################

                self.logger.info('[ Training ] Epoch: {:3d} iteration: {:3d} Loss: [G: {:.3f}] [D: {:.3f}] [G1: {:.3f}] [G2: {:.3f}] [D1: {:.3f}] [D2: {:.3f}]'.format(
                    epoch, i+1, G_loss.item(), D_loss.item(), G1_loss.item(), G2_loss.item(), D1_loss.item(), D2_loss.item()
                ))
            if self.valid_ratio:
                with torch.no_grad():
                    self.G1.eval()
                    self.G2.eval()
                    valid_mse_loss = 0
                    for j, (valid_shadow_img, valid_shadow_free_img, valid_mask_img) in enumerate(self.valid_loader):
                        valid_shadow_img = valid_shadow_img.to(self.device)
                        valid_shadow_free_img = valid_shadow_free_img.to(self.device)
                        valid_mask_img = valid_mask_img.to(self.device)

                        valid_mask_fake = self.G1(valid_shadow_img)
                        valid_shadow_free_fake = self.G2(torch.cat((valid_shadow_img, valid_mask_fake), 1))
                        

                        valid_mse_loss += self.mse_loss(valid_shadow_free_fake, valid_shadow_free_img).item()

                        if j == 0:
                            # save image

                            valid_name = 'validation_{}.png'.format(epoch)
                            # self.logger.info(valid_shadow_free_img.data.cpu().numpy().shape)
                            # self.logger.info(valid_shadow_free_fake.data.cpu().numpy().shape)
                            # self.logger.info(type(valid_mask_fake))
                            # self.logger.info(type(valid_mask_fake.data))
                            # self.logger.info(valid_mask_fake.data)
                            # self.logger.info(valid_mask_fake.data.shape)
                            # valid_mask_fake.data.cpu().numpy()[0, :, :, :]
                            # valid_mask_fake = np.transpose(valid_mask_fake, (1, 2, 0))
                            cv2.imwrite(os.path.join(self.path.valid_gt_shadow_dir, valid_name), np.transpose((valid_shadow_img.data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8), [1, 2, 0]))
                            cv2.imwrite(os.path.join(self.path.valid_gt_shadow_free_dir, valid_name), np.transpose((valid_shadow_free_img.data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8), [1, 2, 0]))
                            cv2.imwrite(os.path.join(self.path.valid_gt_mask_dir, valid_name), np.transpose((valid_mask_img.data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8), [1, 2, 0]))
                            cv2.imwrite(os.path.join(self.path.valid_shadow_free_dir, valid_name), np.transpose((valid_shadow_free_fake.data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8), [1, 2, 0]))
                            cv2.imwrite(os.path.join(self.path.valid_mask_dir, valid_name), np.transpose((valid_mask_fake.data.cpu().numpy()[0, :, :, :] * 255).astype(np.uint8), [1, 2, 0]))
                            

                    valid_mse_loss /= len(self.valid_loader.dataset)
                    self.logger.info('[ Validation ] Epoch: {:3d}, mse loss: {:.3f}'.format(epoch, valid_mse_loss))
                        
                    # self.path.result_dir
                    # mask_eval = self.G1(shadow_img)
                    # shadow_free_eval = self.G2()
            if (epoch+1) % 10 == 0 or epoch == self.epoch - 1:
                self.save(epoch)
    def test(self):
        pass
    def save(self, epoch):
        self.logger.info('Saving model at epoch: {}'.format(epoch))
        torch.save(self.G1.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_G1.ckpt'))
        torch.save(self.G2.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_G2.ckpt'))
        torch.save(self.D1.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_D1.ckpt'))
        torch.save(self.D2.state_dict(), os.path.join(self.mdl_dir, self.mdl_name + '_D2.ckpt'))
        writePickle(self.train_hist, os.path.join(self.mdl_dir, self.mdl_name + '_history.pkl'))
    def load(self):
        self.logger.info('Loading model')
        self.G1.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_G1.ckpt')))
        self.G2.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_G2.ckpt')))
        self.D1.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_D1.ckpt')))
        self.D2.load_state_dict(torch.load(os.path.join(self.mdl_dir, self.mdl_name + '_D2.ckpt')))
        self.train_hist = readPickle(os.path.join(self.mdl_dir, self.mdl_name + '_history.pkl'))
