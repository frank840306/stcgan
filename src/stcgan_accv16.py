import os
import cv2
import sys
import time
import glob
import random
import numpy as np
from tqdm import trange
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import utils

# from dataset import ShadowRemovalDataset, get_composed_transform
from dataset_accv import ShadowRemovalDataset, get_composed_transform

from eval import MetricType, eval_func, eval_dir_func
from fileio import writePickle, readPickle
from logHandler import get_logger

from resource_pix2pix import networks

class STCGAN_ACCV16():
    def __init__(self, args, path):
        self.logger = get_logger(__name__)
        self.mode = args.mode
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.batch_size_test = args.batch_size_test
        # self.valid_ratio = args.valid_ratio
        self.hist_step = args.hist_step
        self.test_step = args.test_step
        self.model_step = args.model_step
        self.start_step = args.start_step
        self.model_name = args.model_name
        self.postfix = args.postfix
        
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
        
        self.device = torch.device('cuda:{}'.format(args.gpu_id))
        # data_loader
        # split data
        if self.mode == 'train' or self.mode == 'test':
            # train_size = len(os.listdir(path.train_shadow_dir))
            # train_img_list = list(range(train_size))
            train_img_list = sorted(os.listdir(path.train_shadow_dir))
            # train_size = len(train_img_list)
            # add augmentation here
            training_transforms = get_composed_transform() if 'randomColor' not in self.postfix else get_composed_transform(aug_dict={
                "RandomHorizontalFlip":{
                    "prob":0.5,
                },
                "RandomVerticalFlip":{
                    "prob": 0.5,
                },
                "RandomRotation":{
                    "min_degree": -60,
                    "max_degree": 60,
                },
                "Resize": {
                    "img_size":[300, 300],
                },
                "RandomCrop":{
                    "img_size":[256, 256],
                },
                "RandomColor": {

                }
            })
            testing_transforms = None 
            train_dataset = ShadowRemovalDataset(path, 'training', train_img_list, training_transforms)
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
            self.test_img_list = sorted(os.listdir(path.test_shadow_dir))
            
            
            test_dataset = ShadowRemovalDataset(path, 'testing', self.test_img_list, testing_transforms)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size_test, shuffle=False, num_workers=8)
            print('Training size: {}, testing size: {}'.format(len(train_dataset), len(test_dataset)))
        elif self.mode == 'infer':
            pass
        else:
            assert(False)
        
        # model
        self.G1 = networks.define_G(input_nc=3, output_nc=1, ngf=64, netG='resnet_6blocks', gpu_ids=[args.gpu_id])
        self.G2 = networks.define_G(input_nc=3+1, output_nc=3, ngf=64, netG='resnet_accv', gpu_ids=[args.gpu_id])
        self.D1 = networks.define_D(input_nc=3+1, ndf=64, netD='n_layers', use_sigmoid=True, gpu_ids=[args.gpu_id])
        self.D2 = networks.define_D(input_nc=3+3+1, ndf=64, netD='n_layers', use_sigmoid=True, gpu_ids=[args.gpu_id])

        self.G_opt = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_opt = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.l1_loss = nn.L1Loss().to(self.device)
        self.gan_loss = networks.GANLoss(use_lsgan=False).to(self.device)


    def train(self):
        self.train_writer = SummaryWriter(os.path.join(self.path.log_dir, 'train'))
        self.test_writer = SummaryWriter(os.path.join(self.path.log_dir, 'test'))
        
        self.D1.train()
        self.D2.train()
        self.logger.info('Start training...')
        start_time = time.time()
        
        batch_num = len(self.train_loader)
        total_steps = self.start_step
        for epoch in range(self.epoch):
            # print('Epoch: {}'.format(epoch + 1))
            self.G1.train()
            self.G2.train()
            # self.logger.info('epoch {}'.format(epoch))
            with trange(len(self.train_loader)) as t:
                for i, (S_img, N_img, M_img, A_img) in enumerate(self.train_loader):
                    # self.logger.info('Iteration: {}'.format(i))
                    # A_real(shadow, S_img), B_real(shadow_free, N_img), C_real(mask, M_img)
                    trainPair = self.set_input(S_img, N_img, M_img, A_img)
                    # self.A_real = shadow_img.to(self.device)
                    # self.B_real = shadow_free_img.to(self.device)
                    # self.C_real = mask_img.to(self.device)

                    trainPair, trainLoss = self.optimize_parameter(trainPair)
                    # sys.stdout.write('[ Training ] Epoch: {:4d}/{:4d} batch: {:3d}/{:3d} Loss: {:.4f} [G: {:.4f}] [D: {:.4f}] [G1: {:.4f}] [G2: {:.4f}] [D1: {:.4f}] [D2: {:.4f}]\r'.format(
                    #     epoch+1, self.epoch, i+1, batch_num, trainLoss['G_loss'] + trainLoss['D_loss'], trainLoss['G_loss'], trainLoss['D_loss'], trainLoss['G1_loss'], trainLoss['G2_loss'], trainLoss['D1_loss'], trainLoss['D2_loss']
                    # ))
                    # self.logger.info('[ Training ] Epoch: {:3d} batch: {:3d} Loss: {:.3f} [G: {:.3f}] [D: {:.3f}] [G1: {:.3f}] [G2: {:.3f}] [D1: {:.3f}] [D2: {:.3f}]'.format(
                    #     epoch, i+1, trainPair['G_loss'] + trainPair['D_loss'], trainPair['G_loss'], trainPair['D_loss'], trainPair['G1_loss'], trainPair['G2_loss'], trainPair['D1_loss'], trainPair['D2_loss']
                    # ))
                    if (total_steps + 1) % self.hist_step == 0:
                        # hist = {}
                        # hist.update(trainLoss)
                        # print(trainLoss)
                        self.record_hist(trainLoss, self.train_writer, total_steps + 1)
                    
                    if (total_steps + 1) % self.test_step == 0:
                        testLoss = self.test(save=False)
                        # hist = {}.update(testLoss)
                        self.record_hist(testLoss, self.test_writer, total_steps + 1)

                    if (total_steps + 1) % self.model_step == 0:
                        self.visualize(total_steps + 1)
                        self.save('latest_{:07d}'.format(total_steps + 1))
                    total_steps += 1
                    t.set_postfix(G1_loss=trainLoss['G1_loss'], G2_loss=trainLoss['G2_loss'], D1_loss=trainLoss['D1_loss'], D2_loss=trainLoss['D2_loss'])
                    t.update()
            print('Iteration: {}'.format(total_steps))
        self.train_writer.close()
        self.test_writer.close()

    def test(self, save=True):
        with torch.no_grad():
            self.G1.eval()
            self.G2.eval()
            loss = {}
            batch_num = len(self.test_loader)
            for i, (S_img, N_img, M_img, A_img) in enumerate(self.test_loader):
                testPair = self.set_input(S_img, N_img, M_img, A_img)
                testPair = self.forward(testPair)
                tmp_loss = {}
                tmp_loss.update(self.calculate_D_loss(testPair))
                tmp_loss.update(self.calculate_G_loss(testPair))
                tmp_loss = {k: v.item() for k, v in tmp_loss.items()}
                loss = dict(Counter(loss) + Counter(tmp_loss))
                
                if save:
                    output_dirs = [self.path.result_shadow_dir, self.path.result_resnet_dir, self.path.result_shadow_free_dir, self.path.result_mask_dir]
                    output_imgs = [testPair['S_real'], testPair['R_fake'], testPair['N_fake'], testPair['M_fake']]
                        
                    for j in range(self.batch_size_test):
                        if i * self.batch_size_test + j == len(self.test_img_list): 
                            break

                        result_name = self.test_img_list[i * self.batch_size_test + j]
                        
                        for output_dir, output_img in zip(output_dirs, output_imgs):
                            fp = os.path.join(output_dir, result_name)
                            img = np.transpose(((output_img.cpu().numpy()[j, :, :, :] + 1) / 2 * 255).astype(np.uint8), [1, 2, 0])
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                            cv2.imwrite(fp, img) 
            loss = {k:v / batch_num for k, v in loss.items()}
            # print(loss)
            if save:
                eval_dir_func(self.path.test_shadow_free_dir, self.path.result_shadow_free_dir, self.path.test_mask_dir, MetricType.MSE | MetricType.RMSE | MetricType.SSIM)
            
        return loss
    def visualize(self, total_steps):
        for i, (S_img, N_img, M_img, A_img) in enumerate(self.test_loader):
            if i == 0:
                testPair = self.set_input(S_img, N_img, M_img, A_img)
                testPair = self.forward(testPair)
                x = vutils.make_grid((testPair['R_fake']+1) / 2, normalize=True, scale_each=True, nrow=4)
                self.test_writer.add_image('Non-shadow', x, total_steps)
                x = vutils.make_grid((testPair['N_fake']+1) / 2, normalize=True, scale_each=True, nrow=4)
                self.test_writer.add_image('Fusion-non-shadow', x, total_steps)
                x = vutils.make_grid((testPair['M_fake']+1) / 2, normalize=True, scale_each=True, nrow=4)
                self.test_writer.add_image('Mask', x, total_steps)
            
            else:
                pass
        return

    def infer(self, fimg):
        with torch.no_grad():
            self.G1.eval()
            self.G2.eval()
            img = cv2.imread(fimg)
            h, w, c = img.shape
            img = cv2.resize(img, (w - w % 32, h - h % 32))
            img = np.transpose(img, (2, 0, 1))
            img = (img.astype(np.float32) / 255 - 0.5) / 0.5
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img)
            img = torch.autograd.Variable(img).cuda()
            out_M = self.G1(img)
            out_N = self.G2(torch.cat((img, out_M), 1))[-1]
            # TODO: infer is incorrect
            out_M = (out_M.cpu().numpy() + 1) / 2 * 255
            out_M = out_M.astype(np.uint8)
            out_M = out_M[-1].transpose((1, 2, 0))

            
            out_N = (out_N.cpu().numpy() + 1) / 2 * 255
            out_N = out_N.astype(np.uint8)
            out_N = out_N.transpose((1, 2, 0))

            if not os.path.exists('out'): os.makedirs('out')
            cv2.imwrite(os.path.join('out', 'mask_' + os.path.basename(fimg)), out_M)
            cv2.imwrite(os.path.join('out', os.path.basename(fimg)), out_N)
            
    
    def optimize_parameter(self, pair):
        pair = self.forward(pair)
        loss = {}
        # TODO: add require grad
        self.D_opt.zero_grad()
        loss.update(self.calculate_D_loss(pair))
        # print('D_loss: {}'.format(loss['D_loss']))
        loss['D_loss'].backward()
        self.D_opt.step()

        # TODO: add require grad
        self.G_opt.zero_grad()
        loss.update(self.calculate_G_loss(pair))
        loss['G_loss'].backward()
        self.G_opt.step()

        loss = {k: v.item() for k, v in loss.items()}
        return pair, loss

        # self.train_hist['loss'].append(self.G_loss.item() + self.D_loss.item())
        
    def set_input(self, S_img, N_img, M_img, A_img):
        # A --> S, B --> N, C --> M
        S_real = S_img.to(self.device)
        N_real = N_img.to(self.device)
        M_real = M_img.to(self.device)
        A_real = A_img.to(self.device)
        
        return {
            'S_real': S_real,
            'N_real': N_real,
            'M_real': M_real,
            'A_real': A_real
        }

    def forward(self, pair):
        pair['M_fake'] = self.G1(pair['S_real'])
        pair['SM_real'] = torch.cat((pair['S_real'], pair['M_real']), 1)
        pair['SM_fake'] = torch.cat((pair['S_real'], pair['M_fake']), 1)
        
        pair['R_fake'], pair['N_fake'] = self.G2(pair['SM_fake'], pair['A_real'])
        pair['SNM_real'] = torch.cat((pair['SM_real'], pair['N_real']), 1)
        pair['SRM_fake'] = torch.cat((pair['SM_fake'], pair['R_fake']), 1)
        pair['SNM_fake'] = torch.cat((pair['SM_fake'], pair['N_fake']), 1)
        # print('G2 output: {}, {}'.format(pair['R_fake'].size(), pair['N_fake'].size()))
        # M_fake is the predicted mask image
        # R_fake is the predicted non-shadow image from original G2
        # N_fake is the fusion output of R_fake and accv16 result 

        return pair


    def calculate_D_loss(self, pair):

        D1_real = self.D1(pair['SM_real'])
        D1_fake = self.D1(pair['SM_fake'].detach())
        # print('D1_real shape: {}\n-------------------'.format(D1_real.size()))
    

        D1_real_loss = self.gan_loss(D1_real, True)
        D1_fake_loss = self.gan_loss(D1_fake, False)
        
        D1_loss = (D1_real_loss * 0.5 + D1_fake_loss * 0.5) * self.lambda2

        D2_real = self.D2(pair['SNM_real'])
        D2_R_fake = self.D2(pair['SRM_fake'].detach())
        D2_N_fake = self.D2(pair['SNM_fake'].detach())


        D2_real_loss = self.gan_loss(D2_real, True)
        D2_R_fake_loss = self.gan_loss(D2_R_fake, False)
        D2_N_fake_loss = self.gan_loss(D2_N_fake, False)
        D2_fake_loss = (D2_R_fake_loss + D2_N_fake_loss) * 0.5

        D2_loss = (D2_real_loss * 0.5 + D2_fake_loss * 0.5) * self.lambda3
        D_loss = D1_loss + D2_loss

        return {
            'D_loss': D_loss,
            'D1_loss': D1_loss,
            'D2_loss': D2_loss,
            'D1_real_loss': D1_real_loss,
            'D1_fake_loss': D1_fake_loss,
            'D2_real_loss': D2_real_loss,
            'D2_fake_loss': D2_fake_loss,
            'D2_R_fake_loss': D2_R_fake_loss,
            'D2_N_fake_loss': D2_N_fake_loss
        }


    def calculate_G_loss(self, pair):

        G1_l1_loss = self.l1_loss(pair['M_fake'], pair['M_real'])
        
        G2_R_l1_loss = self.l1_loss(pair['R_fake'], pair['N_real'])
        G2_N_l1_loss = self.l1_loss(pair['N_fake'], pair['N_real'])
        G2_l1_loss = (G2_R_l1_loss + G2_N_l1_loss) * 0.5

        D1_fake = self.D1(pair['SM_fake'])
        D2_R_fake = self.D2(pair['SRM_fake'])
        D2_N_fake = self.D2(pair['SNM_fake'])
        

        G1_gan_loss = self.gan_loss(D1_fake, True)
        G2_R_gan_loss = self.gan_loss(D2_R_fake, True)
        G2_N_gan_loss = self.gan_loss(D2_N_fake, True)
        G2_gan_loss = (G2_R_gan_loss + G2_N_gan_loss) * 0.5

        G1_loss = G1_l1_loss + G1_gan_loss * self.lambda2
        G2_loss = G2_l1_loss * self.lambda1 + G2_gan_loss * self.lambda3

        G_loss = G1_loss + G2_loss

        return {
            'G_loss': G_loss,
            'G1_loss': G1_loss,
            'G2_loss': G2_loss,
            'G1_l1_loss': G1_l1_loss,
            'G2_l1_loss': G2_l1_loss,
            'G1_gan_loss': G1_gan_loss,
            'G2_gan_loss': G2_gan_loss,
            'G2_R_l1_loss': G2_R_l1_loss,
            'G2_N_l1_loss': G2_N_l1_loss,
            'G2_R_gan_loss': G2_R_gan_loss,
            'G2_N_gan_loss': G2_N_gan_loss
        }
    
    def record_hist(self, hist, writer, total_steps):
        # print(hist)
        for name, value in hist.items():
            writer.add_scalar(name, value, total_steps)

    def save(self, name):
        # self.logger.info('Saving model: {}'.format(name))
        torch.save(self.G1.state_dict(), os.path.join(self.mdl_dir, name + '_G1.ckpt'))
        torch.save(self.G2.state_dict(), os.path.join(self.mdl_dir, name + '_G2.ckpt'))
        torch.save(self.D1.state_dict(), os.path.join(self.mdl_dir, name + '_D1.ckpt'))
        torch.save(self.D2.state_dict(), os.path.join(self.mdl_dir, name + '_D2.ckpt'))
        # writePickle(self.train_hist, os.path.join(self.mdl_dir, self.__class__.__name__ + '_history.pkl'))
    
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
        # self.train_hist = readPickle(os.path.join(self.mdl_dir, self.__class__.__name__ + '_history.pkl'))
