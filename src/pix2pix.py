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

from dataset_pix2pix import ShadowRemovalDataset, get_composed_transform
from eval import MetricType, eval_func, eval_dir_func
from fileio import writePickle, readPickle
from logHandler import get_logger

from resource_pix2pix import networks

class Pix2pix():
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
        self.G = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_6blocks', gpu_ids=[args.gpu_id])
        self.D = networks.define_D(input_nc=3+3, ndf=64, netD='n_layers', use_sigmoid=True, gpu_ids=[args.gpu_id])
        
        self.G_opt = optim.Adam(list(self.G.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_opt = optim.Adam(list(self.D.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.l1_loss = nn.L1Loss().to(self.device)
        self.gan_loss = networks.GANLoss(use_lsgan=False).to(self.device)        

    def train(self):
        self.train_writer = SummaryWriter(os.path.join(self.path.log_dir, 'train'))
        self.test_writer = SummaryWriter(os.path.join(self.path.log_dir, 'test'))
        
        self.D.train()
        self.logger.info('Start training...from {} iteration'.format(self.start_step))
        start_time = time.time()
        
        batch_num = len(self.train_loader)
        total_steps = self.start_step
        for epoch in range(self.epoch):
            self.G.train()
            # self.logger.info('epoch {}'.format(epoch))
            with trange(len(self.train_loader)) as t:
                for i, (S_img, N_img) in enumerate(self.train_loader):
                    # self.logger.info('Iteration: {}'.format(i))
                    # A_real(shadow, S_img), B_real(shadow_free, N_img), C_real(mask, M_img)
                    trainPair = self.set_input(S_img, N_img)
                    
                    trainPair, trainLoss = self.optimize_parameter(trainPair)
                    # sys.stdout.write('[ Training ] Epoch: {:4d}/{:4d} batch: {:3d}/{:3d} Loss: {:.4f} [G: {:.4f}] [D: {:.4f}] [G1: {:.4f}] [G2: {:.4f}] [D1: {:.4f}] [D2: {:.4f}]\r'.format(
                    #     epoch+1, self.epoch, i+1, batch_num, trainLoss['G_loss'] + trainLoss['D_loss'], trainLoss['G_loss'], trainLoss['D_loss'], trainLoss['G1_loss'], trainLoss['G2_loss'], trainLoss['D1_loss'], trainLoss['D2_loss']
                    # ))
                    # self.logger.info('[ Training ] Epoch: {:3d} batch: {:3d} Loss: {:.3f} [G: {:.3f}] [D: {:.3f}] [G1: {:.3f}] [G2: {:.3f}] [D1: {:.3f}] [D2: {:.3f}]'.format(
                    #     epoch, i+1, trainPair['G_loss'] + trainPair['D_loss'], trainPair['G_loss'], trainPair['D_loss'], trainPair['G1_loss'], trainPair['G2_loss'], trainPair['D1_loss'], trainPair['D2_loss']
                    # ))
                    if (total_steps + 1) % self.hist_step == 0:
                        self.record_hist(trainLoss, self.train_writer, total_steps + 1)
                    
                    if (total_steps + 1) % self.test_step == 0:
                        testLoss = self.test(save=False)
                        self.record_hist(testLoss, self.test_writer, total_steps + 1)

                    if (total_steps + 1) % self.model_step == 0:
                        self.visualize(total_steps + 1)
                        self.save('latest_{:07d}'.format(total_steps + 1))
                    total_steps += 1
                    t.set_postfix(G_loss=trainLoss['G_loss'], D_loss=trainLoss['D_loss'])
                    t.update()
            print('Iteration: {}, epoch : {} / {}'.format(total_steps, epoch + 1, self.epoch))
        self.train_writer.close()
        self.test_writer.close()

    def test(self, save=True):
        with torch.no_grad():
            self.G.eval()
            loss = {}
            batch_num = len(self.test_loader)
            for i, (S_img, N_img) in enumerate(self.test_loader):
                testPair = self.set_input(S_img, N_img)
                testPair = self.forward(testPair)
                tmp_loss = {}
                tmp_loss.update(self.calculate_D_loss(testPair))
                tmp_loss.update(self.calculate_G_loss(testPair))
                tmp_loss = {k: v.item() for k, v in tmp_loss.items()}
                loss = dict(Counter(loss) + Counter(tmp_loss))
                
                if save:
                    output_dirs = [self.path.result_shadow_dir, self.path.result_shadow_free_dir]
                    output_imgs = [testPair['S_real'], testPair['N_fake']]
                        
                    for j in range(self.batch_size_test):
                        if i * self.batch_size_test + j == len(self.test_img_list): 
                            break

                        result_name = self.test_img_list[i * self.batch_size_test + j]
                        
                        for output_dir, output_img in zip(output_dirs, output_imgs):
                            fp = os.path.join(output_dir, result_name)
                            img = np.transpose(((output_img.cpu().numpy()[j, :, :, :] + 1) / 2 * 255).astype(np.uint8), [1, 2, 0])
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
                            cv2.imwrite(fp, img) 
            loss = {k:v / batch_num for k, v in loss.items()}
            # print(loss)
            if save:
                eval_dir_func(self.path.test_shadow_free_dir, self.path.result_shadow_free_dir, self.path.test_mask_dir, MetricType.MSE | MetricType.RMSE | MetricType.SSIM)
            
        return loss
    def visualize(self, total_steps):
        for i, (S_img, N_img) in enumerate(self.test_loader):
            if i == 0:
                testPair = self.set_input(S_img, N_img)
                testPair = self.forward(testPair)
                x = vutils.make_grid((testPair['N_fake']+1) / 2, normalize=True, scale_each=True, nrow=4)
                self.test_writer.add_image('Non-shadow', x, total_steps)
            else:
                pass
        return

    def infer(self, fimg):
        print('infer')
        with torch.no_grad():
            self.G.eval()
            img = cv2.imread(fimg)
            print('cv read')
            h, w, c = img.shape
            img = cv2.resize(img, (w - w % 32, h - h % 32))
            img = np.transpose(img, (2, 0, 1))
            img = (img.astype(np.float32) / 255 - 0.5) / 0.5
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img)
            img = torch.autograd.Variable(img).cuda()
            print('before run model')
            out_N = self.G(img)
            
            out_N = (out_N.cpu().numpy() + 1) / 2 * 255
            out_N = out_N.astype(np.uint8)
            print('out: {}'.format(out_N[0].shape))
            out_N = out_N[0].transpose((1, 2, 0))

            if not os.path.exists('out'): os.makedirs('out')
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
        
    def set_input(self, S_img, N_img):
        # A --> S, B --> N, C --> M
        S_real = S_img.to(self.device)
        N_real = N_img.to(self.device)
        return {
            'S_real': S_real,
            'N_real': N_real,
        }

    def forward(self, pair):
        pair['N_fake'] = self.G(pair['S_real'])
        
        pair['SN_real'] = torch.cat((pair['S_real'], pair['N_real']), 1)
        pair['SN_fake'] = torch.cat((pair['S_real'], pair['N_fake']), 1)
        
        return pair

    def calculate_D_loss(self, pair):

        D_real = self.D(pair['SN_real'])
        D_fake = self.D(pair['SN_fake'].detach())
        # print('D1_real shape: {}\n-------------------'.format(D1_real.size()))
    

        D_real_loss = self.gan_loss(D_real, True) * 0.5
        D_fake_loss = self.gan_loss(D_fake, False) * 0.5
        
        D_loss = (D_real_loss + D_fake_loss) * self.lambda2

        return {
            'D_loss': D_loss,
            'D_real_loss': D_real_loss,
            'D_fake_loss': D_fake_loss,
        }


    def calculate_G_loss(self, pair):

        G_l1_loss = self.l1_loss(pair['N_fake'], pair['N_real'])

        D_fake = self.D(pair['SN_fake'])
        
        G_gan_loss = self.gan_loss(D_fake, True)
        
        G_loss = G_l1_loss + G_gan_loss * self.lambda2
        
        return {
            'G_loss': G_loss,
            'G_l1_loss': G_l1_loss,
            'G_gan_loss': G_gan_loss,
            
        }
    
    def record_hist(self, hist, writer, total_steps):
        # print(hist)
        for name, value in hist.items():
            writer.add_scalar(name, value, total_steps)

    def save(self, name):
        # self.logger.info('Saving model: {}'.format(name))
        torch.save(self.G.state_dict(), os.path.join(self.mdl_dir, name + '_G.ckpt'))
        torch.save(self.D.state_dict(), os.path.join(self.mdl_dir, name + '_D.ckpt'))
    
    def load(self):
        # name, best or latest
        
        if self.model_name == 'latest':
            # load latest
            mdl = sorted(glob.glob(os.path.join(self.mdl_dir, 'latest*')))[-1].split('/')[-1][:-7]
            
        elif self.model_name == 'best':
            # load best
            mdl = sorted(glob.glob(os.path.join(self.mdl_dir, 'best*')))[-1].split('/')[-1][:-7]
            
        else:
            # load by name
            mdl = self.model_name
        
        self.logger.info('Loading model: {}'.format(mdl))
        self.G.load_state_dict(torch.load(os.path.join(self.mdl_dir, mdl + '_G.ckpt')))
        self.D.load_state_dict(torch.load(os.path.join(self.mdl_dir, mdl + '_D.ckpt')))
