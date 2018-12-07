import os
import cv2
import numpy as np
from enum import IntEnum
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from logHandler import set_logger, get_logger


class MetricType(IntEnum):
    MSE = 2**0
    RMSE = 2**1
    BER = 2**2
    PSNR = 2**3
    SSIM = 2**4

    ALL = 2**5 - 1

PIXEL_MAX = 255

default_score_dict = {
    'mse': {
        'all': 0,
        'shadow': 0,
        'shadow_free': 0,
    },
    'rmse': {
        'all': 0,
        'shadow': 0,
        'shadow_free': 0,
    }, 
    'psnr': {
        'all': 0,
        'shadow': 0,
        'shadow_free': 0,
    },
    'ssim': {
        'all': 0,
        'shadow': 0,
        'shadow_free': 0,
    }
}

def mse_score(img1, img2, pixel_num=None):  
    if pixel_num:
        scores = np.sum((img1 - img2) ** 2) / pixel_num
    else:
        scores = np.mean((img1 - img2) ** 2)
    return scores

def rmse_score(img1, img2, pixel_num=None):
    scores = np.sqrt(mse_score(img1, img2, pixel_num))
    return scores

def eval_func(ftrue, ftest, fmask=None, metrics=MetricType.ALL):
    # metrics 
    scores = deepcopy(default_score_dict)
    # print(ftrue, ftest)
    all_true_img = cv2.imread(ftrue)
    all_test_img = cv2.imread(ftest)
    all_true_img = cv2.cvtColor(all_true_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    all_test_img = cv2.cvtColor(all_test_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    pixel_types = ['all', ]
    true_imgs = [all_true_img, ]
    test_imgs = [all_test_img, ]
    pixel_nums = [None, ]

    if fmask:
        mask_img = cv2.imread(fmask)
        mask_img = mask_img != 0
        shadow_pixel = np.sum(mask_img)
        shadow_free_pixel = mask_img.shape[0] * mask_img.shape[1] * mask_img.shape[2] - shadow_pixel
        shadow_true_img = np.multiply(all_true_img, mask_img==1)
        shadow_free_true_img = np.multiply(all_true_img, mask_img==0)
        shadow_test_img = np.multiply(all_test_img, mask_img==1)
        shadow_free_test_img = np.multiply(all_test_img, mask_img==0)

        pixel_types += ['shadow', 'shadow_free']
        true_imgs += [shadow_true_img, shadow_free_true_img]
        test_imgs += [shadow_test_img, shadow_free_test_img]
        pixel_nums += [shadow_pixel, shadow_free_pixel]

    for pixel_type, true_img, test_img, pixel_num in zip(pixel_types, true_imgs, test_imgs, pixel_nums): 
        if metrics & MetricType.MSE:
            scores['mse'][pixel_type] = mse_score(true_img, test_img, pixel_num)
        if metrics & MetricType.RMSE:
            scores['rmse'][pixel_type] = rmse_score(true_img, test_img, pixel_num)
    return scores

def eval_dir_func(true_dir, test_dir, mask_dir=None, metrics=MetricType.ALL):
    logger = get_logger(__name__)
    avg_scores = deepcopy(default_score_dict)
    flist = os.listdir(true_dir)
    file_num = len(flist)
    logger.info('Eval image num: {}\nTrue img: [ {} ]\nTest img: [ {} ]\nMask img: [ {} ]'.format(file_num, true_dir, test_dir, mask_dir))
    pixel_types = ['all']
    if mask_dir:
        pixel_types += ['shadow', 'shadow_free']
    for fname in flist:
        ftrue = os.path.join(true_dir, fname)
        ftest = os.path.join(test_dir, fname)
        if mask_dir:
            fmask = os.path.join(mask_dir, fname)
            scores = eval_func(ftrue, ftest, fmask, metrics)
        else:
            scores = eval_func(ftrue, ftest, None, metrics)
        # print(scores)
        for pixel_type in pixel_types:
            if metrics & MetricType.MSE: 
                avg_scores['mse'][pixel_type] += scores['mse'][pixel_type] / file_num
            if metrics & MetricType.RMSE: 
                avg_scores['rmse'][pixel_type] += scores['rmse'][pixel_type] / file_num
    logger.info(avg_scores)
    return avg_scores

if __name__ == '__main__':
    log_file = os.path.join('log', os.path.basename(__file__) + '.log')
    set_logger(log_file)
    logger = get_logger(__name__)

    test_metrics = MetricType.MSE | MetricType.RMSE

    true_dir = os.path.join('processed_dataset', 'ISTD', 'test', 'non_shadow')
    mask_dir = os.path.join('processed_dataset', 'ISTD', 'test', 'mask')
    guo_dir = os.path.join('processed_dataset', 'ISTD', 'result', 'Guo')
    yang_dir = os.path.join('processed_dataset', 'ISTD', 'result', 'Yang')
    gong_dir = os.path.join('processed_dataset', 'ISTD', 'result', 'gong')
    stcgan_dir = os.path.join('processed_dataset', 'ISTD', 'result', 'ST-CGAN')
    
    avg_eval_scores = eval_dir_func(true_dir, guo_dir, mask_dir, test_metrics)
    avg_eval_scores = eval_dir_func(true_dir, yang_dir, mask_dir, test_metrics)
    avg_eval_scores = eval_dir_func(true_dir, gong_dir, mask_dir, test_metrics)
    avg_eval_scores = eval_dir_func(true_dir, stcgan_dir, mask_dir, test_metrics)
    
    