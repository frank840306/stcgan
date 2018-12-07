
import argparse
import os
import utils
import random
import torch
import preprocess_dataset
# from stcgan import STCGAN
from stcgan_from_pix2pix import STCGAN
from logHandler import get_logger, set_logger
from pathHandler import PathHandler




def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str, help='train or test')
    parser.add_argument('--epochs', default=100, type=int, help='# of epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='# of batch size')
    parser.add_argument('--lrG', default=0.1, type=float, help='learning rate of G')
    parser.add_argument('--lrD', default=0.1, type=float, help='learning rate of D')
    parser.add_argument('--lambda1', default=5, type=float, help='for data2 loss usage')
    parser.add_argument('--lambda2', default=0.1, type=float, help='for gan1 loss usage')
    parser.add_argument('--lambda3', default=0.1, type=float, help='for gan2 loss usage')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 of adam opt')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam opt')
    parser.add_argument('--valid_ratio', default=0.01, type=float, help='validation data ratio')
    parser.add_argument('--gpu_mode', default=True, type=bool, help='use gpu or not')
    parser.add_argument('--gpu_id', default=1, type=int, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load', default=False, type=bool, help='load pretrained model or not')
    parser.add_argument('--root_dir', default='.', type=str, help='the root folder')
    parser.add_argument('--seed', default=840306, type=int, help='random seed')
    # parser.add_argument()
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    path = PathHandler(args.root_dir)

    # TODO: use small train #####################################
    # print('use small train dataset!!!')
    # path.train_dir = os.path.join(path.data_dir, 'small_train')
    # path.train_mask_dir = os.path.join(path.train_dir, 'mask')
    # path.train_shadow_dir = os.path.join(path.train_dir, 'shadow')
    # path.train_shadow_free_dir = os.path.join(path.train_dir, 'non_shadow')
    # !!!!!!!!!!!!!!!!!@@@@@@@@@@%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^


    log_file = os.path.join(path.log_dir, os.path.basename(__file__) + '.log')
    set_logger(log_file)
    logger = get_logger(__name__)
    logger.info(args)
    # setting random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    net = STCGAN(args, path)
    
    if args.load:
        net.load()
    
    net.train()

        