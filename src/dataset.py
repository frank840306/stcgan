import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

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
        img_id = self.img_list[idx]
        
        if self.data_type == 'training' or self.data_type == 'validation':
            datatype_dir = self.path.train_dir
        elif self.data_type == 'testing':
            datatype_dir = self.path.test_dir
        else: 
            self.logger.error('Unexpected data type: {}'.format(self.data_type))
        shadow_img = cv2.imread(os.path.join(datatype_dir, 'shadow', '{:0>6d}.png'.format(img_id)))
        shadow_free_img = cv2.imread(os.path.join(datatype_dir, 'non_shadow', '{:0>6d}.png'.format(img_id)))
        mask_img = cv2.imread(os.path.join(datatype_dir, 'mask', '{:0>6d}.png'.format(img_id)), cv2.IMREAD_GRAYSCALE)
        
        # TODO: resize
        if self.data_type != 'testing': 
            pass
        # TODO: add augmentation
        # print(self.transform)
        if self.transform:
            shadow_img, shadow_free_img, mask_img = self.transform(shadow_img, shadow_free_img, mask_img)
        
        shadow_img = np.transpose(shadow_img, (2, 0, 1))
        shadow_free_img = np.transpose(shadow_free_img, (2, 0, 1))
        mask_img = mask_img[np.newaxis, :, :]
        # CHW
        return torch.from_numpy(shadow_img).float()/255, torch.from_numpy(shadow_free_img).float()/255, torch.from_numpy(mask_img).float()/255

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
            "Resize": {
                "img_size":[300, 300],
            },
            "RandomCrop":{
                "img_size":[256, 256],
            }
        }
    for aug_key, aug_param in sorted(aug_dict.items(), reverse=True):
        augmentations.append(key2aug[aug_key](**aug_param))
    return Compose(augmentations)



class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
    def __call__(self, shadow_img, shadow_free_img, mask_img):
        
        for aug in self.augmentations:
            shadow_img, shadow_free_img, mask_img = aug(shadow_img, shadow_free_img, mask_img)
        
        return shadow_img, shadow_free_img, mask_img
    
class Resize(object):
    def __init__(self, img_size):
        self.height, self.width = img_size
    def __call__(self, shadow_img, shadow_free_img, mask_img):
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
        
class RandomCrop(object):
    def __init__(self, img_size):
        self.height, self.width = img_size

    def __call__(self, shadow_img, shadow_free_img, mask_img):
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

key2aug = {
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'Resize': Resize,
    'RandomCrop': RandomCrop,
    
}