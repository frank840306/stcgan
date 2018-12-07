import os
import argparse

from pathHandler import PathHandler
from logHandler import get_logger, set_logger

from stcgan_pix2pix import STCGAN

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', type=str, help='train or test')
    parser.add_argument('--epochs', default=500, type=int, help='# of epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='# of batch size when train')
    parser.add_argument('--batch_size_test', default=2, type=int, help='# of batch size when valid and test')
    parser.add_argument('--lrG', default=0.01, type=float, help='learning rate of G')
    parser.add_argument('--lrD', default=0.01, type=float, help='learning rate of D')
    parser.add_argument('--lambda1', default=5, type=float, help='for G2 loss usage')
    parser.add_argument('--lambda2', default=0.1, type=float, help='for D1 loss usage')
    parser.add_argument('--lambda3', default=0.1, type=float, help='for D2 loss usage')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 of adam opt')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam opt')
    parser.add_argument('--valid_ratio', default=0.05, type=float, help='validation data ratio')
    # parser.add_argument('--gpu_mode', default=True, type=bool, help='use gpu or not')
    parser.add_argument('--gpu_id', default=1, type=int, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_model', action='store_true', help='load pretrained model or not')
    parser.add_argument('--root_dir', default='.', type=str, help='the root folder')
    parser.add_argument('--manual_seed', default=840306, type=int, help='random seed')
    parser.add_argument('--valid_step', default=30, type=int, help='every # steps to validation')
    parser.add_argument('--model_step', default=10, type=int, help='every # epoch save model')
    parser.add_argument('--model_name', default='latest', type=str, help='load the model, latest, besr or name')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    path = PathHandler(root_dir=args.root_dir)

    log_file = os.path.join(path.log_dir, os.path.basename(__file__) + '.log')
    set_logger(log_file)
    logger = get_logger(__name__)
    logger.info(args)
    
    net = STCGAN(args, path)
    if args.load_model:
        net.load()

    if args.mode == 'train':
        net.train()
    elif args.mode == 'test':
        net.test()
    else:
        logger.info('Unexpected mode: {}'.format(args.mode))
