import os
import glob
import time
import argparse
import numpy as np

from pathHandler import PathHandler
from logHandler import get_logger, set_logger

from stcgan_pix2pix import STCGAN
from stcgan_fusion import STCGAN_ACCV16
# from stcganGlobalHist import StcganGlobalHist

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str, help='train or test')
    parser.add_argument('--epochs', default=500, type=int, help='# of epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='# of batch size when train')
    parser.add_argument('--batch_size_test', default=8, type=int, help='# of batch size when valid and test')
    parser.add_argument('--lrG', default=0.001, type=float, help='learning rate of G')
    parser.add_argument('--lrD', default=0.001, type=float, help='learning rate of D')
    parser.add_argument('--lambda1', default=5, type=float, help='for G2 loss usage')
    parser.add_argument('--lambda2', default=0.1, type=float, help='for D1 loss usage')
    parser.add_argument('--lambda3', default=0.1, type=float, help='for D2 loss usage')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 of adam opt')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam opt')
    # parser.add_argument('--valid_ratio', default=0.0, type=float, help='validation data ratio') # 0.05
    # parser.add_argument('--gpu_mode', default=True, type=bool, help='use gpu or not')
    parser.add_argument('--gpu_id', default=0, type=int, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_model', action='store_true', help='load pretrained model or not')
    parser.add_argument('--root_dir', default='.', type=str, help='the root folder')
    parser.add_argument('--task_name', default='stcgan', type=str, help='task name')
    parser.add_argument('--postfix', default='', type=str, help='postfix of task name')
    parser.add_argument('--manual_seed', default=840306, type=int, help='random seed')
    parser.add_argument('--hist_step', default=20, type=int, help='every # steps record training statistic')
    parser.add_argument('--test_step', default=200, type=int, help='every # steps to validation')
    parser.add_argument('--model_step', default=2000, type=int, help='every # steps save model')
    parser.add_argument('--start_step', default=0, type=int, help='start step')
    parser.add_argument('--model_name', default='latest', type=str, help='load the model, latest, besr or name')
    parser.add_argument('--dataset_name', default='ISTD', type=str, help='dataset name, ISTD or DSRD')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    path = PathHandler(root_dir=args.root_dir, task='{}_lrG_{}_lrD_{}{}'.format(args.task_name, args.lrG, args.lrD, args.postfix), data=args.dataset_name)
    #path.test_dir = './src/document'
    #path.test_shadow_dir = './src/document/shadow'
    log_file = os.path.join(path.log_dir, os.path.basename(__file__) + '.log')
    set_logger(log_file)
    logger = get_logger(__name__)
    logger.info(args)
    
    # start_time = time.time()
    if args.task_name == 'stcgan':
        net = STCGAN(args, path)
    elif args.task_name == 'stcgan_fusion':
        net = STCGAN_ACCV16(args, path)
    # end_time = time.time()
    # print('Time of initializing model: {} sec'.format(end_time - start_time))

    if args.load_model:
        # start_time = time.time()
        net.load()
        # end_time = time.time()
        # print('Time of loading model: {} sec'.format(end_time - start_time))

    if args.mode == 'train':
        net.train()
    elif args.mode == 'test':
        net.test()
    elif args.mode == 'infer':
        exts = ['png', 'PNG', 'jpg', 'JPG']
        # exec_times = []
        for fname in sorted(os.listdir('demo')):
            for ext in exts:
                if fname.endswith(ext):
                    # start_time = time.time()
                    net.infer(os.path.join('demo', fname))
                    # end_time = time.time()
                    # exec_times.append(end_time - start_time)
        # print('Average time of predicting single image: {} sec'.format(np.mean(exec_times)))
    else:
        logger.info('Unexpected mode: {}'.format(args.mode))
