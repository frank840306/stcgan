import os

from rmse import rmse_score


def eval():
    root_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.join(root_dir, 'dataset', 'UIUC', 'cleaned_training', 'data', 'Our_test')
    print('dataset directory: {}'.format(dataset_dir))
    # rmse score
    fname = os.path.join(dataset_dir, 'wholelist.txt')
    

if __name__ == '__main__':
    eval()