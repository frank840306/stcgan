import os
import cv2
import random
import numpy as np
import imutils
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from scipy.interpolate import UnivariateSpline
from skimage import transform



from logHandler import get_logger

class ShadowRemovalDataset(Dataset):
    def __init__(self, path=None, data_type='training', img_list=None, transform=None):
        self.logger = get_logger(__name__)
        if not path:
            from pathHandler import PathHandler
            self.path = PathHandler()
        else:
            self.path = path
        self.data_type = data_type
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        
        if self.data_type == 'training' or self.data_type == 'validation':
            datatype_dir = self.path.train_dir
        elif self.data_type == 'testing':
            datatype_dir = self.path.test_dir
        else: 
            self.logger.error('Unexpected data type: {}'.format(self.data_type))
        shadow_img = cv2.imread(os.path.join(datatype_dir, 'shadow', img_name))
        shadow_free_img = cv2.imread(os.path.join(datatype_dir, 'non_shadow', img_name))
        mask_img = cv2.imread(os.path.join(datatype_dir, 'mask', img_name), cv2.IMREAD_GRAYSCALE)

        shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)
        shadow_free_img = cv2.cvtColor(shadow_free_img, cv2.COLOR_BGR2RGB)
        
        
        # print(os.path.join(datatype_dir, 'shadow', '{:0>6d}.png'.format(img_name)))
        # print(os.path.join(datatype_dir, 'non_shadow', '{:0>6d}.png'.format(img_name)))
        # TODO: resize
        if self.data_type != 'testing': 
            pass
        # TODO: add augmentation
        # print(self.transform)
        if self.transform:
            shadow_img, shadow_free_img, mask_img = self.transform(shadow_img, shadow_free_img, mask_img)
        
        # shadow_img = np.transpose(shadow_img, (2, 0, 1)).astype(np.float32)
        # shadow_free_img = np.transpose(shadow_free_img, (2, 0, 1)).astype(np.float32)
        mask_img = mask_img[:, :, np.newaxis]
        
        tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        #print(shadow_img.shape, shadow_free_img.shape, mask_img.shape)
        shadow_img = tfs(shadow_img)
        shadow_free_img = tfs(shadow_free_img)
        mask_img = tfs(mask_img)
        # print(shadow_img.size())
        return shadow_img, shadow_free_img, mask_img

        # CHW
        # return torch.from_numpy(shadow_img).float()/255, torch.from_numpy(shadow_free_img).float()/255, torch.from_numpy(mask_img).float()/255

    def __len__(self):
        return len(self.img_list)
    

# TODO: add augmentation
def get_composed_transform(aug_dict=None):
    augmentations = []
    if not aug_dict:
        aug_dict = {
            "RandomHorizontalFlip":{
                "prob":0.5,
            },
            "RandomVerticalFlip":{
                "prob": 0.5,
            },
            'RandomRotation':{
                "min_degree": -60,
                "max_degree": 60,
            },
            "Resize": {
                "img_size":[300, 300],
            },
            "RandomCrop":{
                "img_size":[256, 256],
            },
            # "RandomColor": {

            # }
        }
    get_logger(__name__).info('Augmentation method: {}'.format(', '.join(aug_dict.keys())))
    for aug_key, aug_param in sorted(aug_dict.items(), reverse=True):
        # print(aug_key)
        augmentations.append(key2aug[aug_key](**aug_param))
    # print('====\n\n')
    return Compose(augmentations)



class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
    def __call__(self, shadow_img, shadow_free_img, mask_img):
        
        for aug in self.augmentations:
            shadow_img, shadow_free_img, mask_img = aug(shadow_img, shadow_free_img, mask_img)
        
        # import matplotlib.pyplot as plt
        # plt.figure('S')
        # plt.imshow(shadow_img)
        # plt.figure('N')
        # plt.imshow(shadow_free_img)
        # plt.figure('M')
        # plt.imshow(mask_img)
        # plt.show()
        return shadow_img, shadow_free_img, mask_img
    
class Resize(object):
    def __init__(self, img_size):
        self.height, self.width = img_size
    def __call__(self, shadow_img, shadow_free_img, mask_img):
        # print('resize')
        assert shadow_img.shape[0] == shadow_free_img.shape[0]
        assert shadow_img.shape[1] == shadow_free_img.shape[1]
        assert shadow_img.shape[0] == mask_img.shape[0]
        assert shadow_img.shape[1] == mask_img.shape[1]
        # print(shadow_free_img.shape[0], self.height)
        assert shadow_free_img.shape[0] >= self.height
        assert shadow_free_img.shape[1] >= self.width
        shadow_img = cv2.resize(shadow_img, (self.width, self.height), cv2.INTER_LINEAR)
        shadow_free_img = cv2.resize(shadow_free_img, (self.width, self.height), cv2.INTER_LINEAR)
        mask_img = cv2.resize(mask_img, (self.width, self.height), cv2.INTER_LINEAR)
        return shadow_img, shadow_free_img, mask_img

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, shadow_img, shadow_free_img, mask_img):
        # print('horizon')
        if random.random() >= self.prob:
            
            assert shadow_img.shape[0] == shadow_free_img.shape[0]
            assert shadow_img.shape[1] == shadow_free_img.shape[1]
            assert shadow_img.shape[0] == mask_img.shape[0]        
            assert shadow_img.shape[1] == mask_img.shape[1]        
            
            # shadow_img = cv2.flip(shadow_img, 0)
            # shadow_free_img = cv2.flip(shadow_free_img, 0)
            shadow_img = np.fliplr(shadow_img).copy()
            shadow_free_img = np.fliplr(shadow_free_img).copy()
            mask_img = np.fliplr(mask_img).copy()
        
        return shadow_img, shadow_free_img, mask_img
        
class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, shadow_img, shadow_free_img, mask_img):
        # print('vertical')
        if random.random() >= self.prob:
            
            assert shadow_img.shape[0] == shadow_free_img.shape[0]
            assert shadow_img.shape[1] == shadow_free_img.shape[1]
            assert shadow_img.shape[0] == mask_img.shape[0]
            assert shadow_img.shape[1] == mask_img.shape[1]
            
            # shadow_img = cv2.flip(shadow_img, 0)
            # shadow_free_img = cv2.flip(shadow_free_img, 0)
            shadow_img = np.flipud(shadow_img).copy()
            shadow_free_img = np.flipud(shadow_free_img).copy()
            mask_img = np.flipud(mask_img).copy()
        
        return shadow_img, shadow_free_img, mask_img

class RandomRotation(object):
    def __init__(self, min_degree, max_degree):
        self.min_degree = min_degree
        self.max_degree = max_degree

    def __call__(self, shadow_img, shadow_free_img, mask_img):
        # print('rotate')
        assert shadow_img.shape[0] == shadow_free_img.shape[0]
        assert shadow_img.shape[1] == shadow_free_img.shape[1]
        assert shadow_img.shape[0] == mask_img.shape[0]
        assert shadow_img.shape[1] == mask_img.shape[1]

        rotate_degree = random.uniform(self.min_degree, self.max_degree)
        shadow_img = imutils.rotate(shadow_img, rotate_degree)
        shadow_free_img = imutils.rotate(shadow_free_img, rotate_degree)
        mask_img = imutils.rotate(mask_img, rotate_degree)
        # shadow_img = transform.rotate(shadow_img, rotate_degree).copy()
        # shadow_free_img = transform.rotate(shadow_free_img, rotate_degree).copy()
        # mask_img = transform.rotate(mask_img, rotate_degree).copy()

        # print(shadow_img.shape, shadow_free_img.shape, mask_img.shape)
    
        return shadow_img, shadow_free_img, mask_img
        

class RandomCrop(object):
    def __init__(self, img_size):
        self.height, self.width = img_size

    def __call__(self, shadow_img, shadow_free_img, mask_img):
        # print('crop')
        # HWC HWC HW
        assert shadow_img.shape[0] == shadow_free_img.shape[0]
        assert shadow_img.shape[1] == shadow_free_img.shape[1]
        assert shadow_img.shape[0] == mask_img.shape[0]
        assert shadow_img.shape[1] == mask_img.shape[1]
        # print(shadow_free_img.shape[0], self.height)
        assert shadow_free_img.shape[0] >= self.height
        assert shadow_free_img.shape[1] >= self.width
        
        x = random.randint(0, mask_img.shape[1] - self.width)
        y = random.randint(0, mask_img.shape[0] - self.height)

        shadow_img = shadow_img[y : y + self.height, x : x + self.width, :]
        shadow_free_img = shadow_free_img[y : y + self.height, x : x + self.width, :]
        mask_img = mask_img[y : y + self.height, x : x + self.width]
        # print('range {} to {}, {}to {}'.format(y, y+self.height, x, x+self.width))
        return shadow_img, shadow_free_img, mask_img

class RandomColor(object):
    def __init__(self):
        incr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 65, 130, 195, 256])     # [0, 70, 140, 210, 256] 
        decr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 60, 120, 165, 220])      # [0, 30, 80, 120, 192]
        self.luts = [incr_ch_lut, decr_ch_lut]
    def create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    def randomParam(self):
        h = random.uniform(-2, 2)
        s = random.uniform(-10, 10)
        v = random.uniform(-10, 10)
        g = random.uniform(0.7, 1.3)
        m = random.choice([0, 1])
        return h, s, v, g, m
    
    def gamma_corretion(self, img, gamma=1):
        gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)    
        gamma_img = cv2.LUT(img,gamma_table)
        return gamma_img

    def adjustHSV(self, img, hue=0, saturation=0, value=0):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        c_h, c_s, c_v = cv2.split(img)

        c_h = np.clip(c_h + hue, 0, 179)
        c_s = np.clip(c_s + saturation, 0, 255)
        c_v = np.clip(c_v + value, 0, 255)

        img = cv2.merge((c_h, c_s, c_v)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # img = np.clip(img, 0, 255)
        return img
    
    def white_balance(self, img, mode=0):
        # warmer : 0, colder : 1
        c_b, c_g, c_r = cv2.split(img)
        c_r = cv2.LUT(c_r, self.luts[mode]).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.luts[mode ^ 1]).astype(np.uint8)
        img = cv2.merge((c_b, c_g, c_r))

        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, self.luts[mode]).astype(np.uint8)
        
        img = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
        return img


    def __call__(self, shadow_img, shadow_free_img, mask_img):
        # HWC HWC HW
        assert shadow_img.shape[0] == shadow_free_img.shape[0]
        assert shadow_img.shape[1] == shadow_free_img.shape[1]
        assert shadow_img.shape[0] == mask_img.shape[0]
        assert shadow_img.shape[1] == mask_img.shape[1]

        shadow_img = shadow_img.copy()
        shadow_free_img = shadow_free_img.copy()

        h, s, v, g, m = self.randomParam()
        # print("XD")        
        shadow_img = self.gamma_corretion(shadow_img, g)
        shadow_free_img = self.gamma_corretion(shadow_free_img, g)

        shadow_img = self.adjustHSV(shadow_img, h, s, v)
        shadow_free_img = self.adjustHSV(shadow_free_img, h, s, v)
        
        shadow_img = self.white_balance(shadow_img, m)
        shadow_free_img = self.white_balance(shadow_free_img, m)
        
        return shadow_img, shadow_free_img, mask_img


key2aug = {
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomRotation': RandomRotation,
    'Resize': Resize,
    'RandomCrop': RandomCrop,
    'RandomColor': RandomColor,
    
}
