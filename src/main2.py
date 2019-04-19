import argparse

# def get_cfg():
#     # about model architecture

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='StcganGlobalHist', type=str, help='')
    parser.add_argument('--root_dir', type=str, help='')
    parser.add_argument('--mode', default='test', type=str, help='train, test or infer')
    parser.add_argument('--load_model', nargs='+', type=int, default=[1, 1, 1], help='whether to load models, histNet, detectNet, removalNet')
    parser.add_argument('--epoch', default=500, type=int, help='')
    parser.add_argument('--train_batch_size', default=8, type=int, help='')
    parser.add_argument('--test_batch_size', default=4, type=int, help='')
    # parser.add_argument('')
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    parser.add_argument('--manual_seed', default=840306, type=int, help='')
    return parser.parse_args()

def main():
    args = get_args()
    print(vars(args))
if __name__ == "__main__":
    main()