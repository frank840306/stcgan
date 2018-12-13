import os
import glob
import argparse

from logHandler import set_logger, get_logger
from pathHandler import PathHandler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str, help='the root folder')
    parser.add_argument('--eval_type', default='all', type=str, help='best, atest or all')
    parser.add_argument('--batch_size_test', default=4, type=int, help='batch size of testing image')
    parser.add_argument('--lrG', default=0.003, type=float, help='lr of G')
    parser.add_argument('--lrD', default=0.003, type=float, help='lr of D')

    return parser.parse_args()

def eval_model(eval_type='best'):
    best_models = glob.glob(os.path.join(path.mdl_dir, '{}*'.format(eval_type)))
    # print(best_models)
    best_models = sorted(list(set([bm.split('/')[-1][:-8] for bm in best_models])))
    for m in best_models:
        # pass
        os.system('./run.sh --load_model --gpu_id 0 --model_name {} --lrG {} --lrD {} --batch_size_test {} --root_dir {}'.format(m, args.lrG, args.lrD, args.batch_size_test, args.root_dir))

def eval_all():
    eval_model('best')
    eval_model('latest')
    
if __name__ == "__main__":
    args = get_args()
    path = PathHandler(args.root_dir, task='stcgan_lrG_{}_lrD_{}'.format(args.lrG, args.lrD))
    
    log_file = os.path.join(path.log_dir, os.path.basename(__file__) + '.log')
    set_logger(log_file)
    logger = get_logger(__name__)
    logger.info(args)
    
    if args.eval_type == 'all':
        eval_all()
    else:
        eval_model(args.eval_type)