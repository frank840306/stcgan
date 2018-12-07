import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from logHandler import set_logger, get_logger
import numpy as np

def check_ISTD(dst_dir):
    # check ISTD dataset
    training_size = 1330
    testing_size = 540
    
    train_dir = os.path.join(dst_dir, 'train')
    test_dir = os.path.join(dst_dir, 'test')

    train_shadow_dir = os.path.join(train_dir, 'shadow')
    train_non_shadow_dir = os.path.join(train_dir, 'non_shadow')
    train_mask_dir = os.path.join(train_dir, 'mask')
    
    test_shadow_dir = os.path.join(test_dir, 'shadow')
    test_non_shadow_dir = os.path.join(test_dir, 'non_shadow')
    test_mask_dir = os.path.join(test_dir, 'mask')

    result_Gong_dir = os.path.join(dst_dir, 'result', 'Gong')
    result_Guo_dir = os.path.join(dst_dir, 'result', 'Guo')
    result_Yang_dir = os.path.join(dst_dir, 'result', 'Yang')
    result_STCGAN_dir = os.path.join(dst_dir, 'result', 'ST-CGAN')

    # check size 
    if not all([
        len(os.listdir(train_shadow_dir)) == training_size,
        len(os.listdir(train_non_shadow_dir)) == training_size,
        len(os.listdir(train_mask_dir)) == training_size,
        len(os.listdir(test_shadow_dir)) == testing_size,
        len(os.listdir(test_non_shadow_dir)) == testing_size,
        len(os.listdir(test_mask_dir)) == testing_size,
        len(os.listdir(result_Gong_dir)) == testing_size,
        len(os.listdir(result_Guo_dir)) == testing_size,
        len(os.listdir(result_Yang_dir)) == testing_size,
        len(os.listdir(result_STCGAN_dir)) == testing_size
    ]):
        logger.error('Size of ISTD dataset is incorrect\n[ {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]'.format(
            len(os.listdir(train_shadow_dir)) == training_size,
            len(os.listdir(train_non_shadow_dir)) == training_size,
            len(os.listdir(train_mask_dir)) == training_size,
            len(os.listdir(test_shadow_dir)) == testing_size,
            len(os.listdir(test_non_shadow_dir)) == testing_size,
            len(os.listdir(test_mask_dir)) == testing_size,
            len(os.listdir(result_Gong_dir)) == testing_size,
            len(os.listdir(result_Guo_dir)) == testing_size,
            len(os.listdir(result_Yang_dir)) == testing_size,
            len(os.listdir(result_STCGAN_dir)) == testing_size
        ))
        return False
    # check training dataset content
    if True:
        sample_num = 10
        sample_idx = np.random.randint(training_size, size=sample_num)
        for idx in sample_idx:
            shadow_filename = os.path.join(train_shadow_dir, '%06d.png' % (idx))
            non_shadow_filename = os.path.join(train_non_shadow_dir, '%06d.png' % (idx))
            mask_filename = os.path.join(train_mask_dir, '%06d.png' % (idx))
            
            logger.info('Show [ {} ], [ {} ] and [ {} ]'.format(shadow_filename, non_shadow_filename, mask_filename))
            
            pos_list = [131, 132, 133]
            title_list = ['shadow', 'non_shadow', 'mask']
            filename_list = [shadow_filename, non_shadow_filename, mask_filename]
            kwargs_list = [{}, {}, {'cmap': 'Greys_r'}]
            
            fig = plt.figure()
            fig.suptitle('Traing data index: {}'.format(idx))
            for pos, title, filename, kwargs in zip(pos_list, title_list, filename_list, kwargs_list):
                ax = fig.add_subplot(pos)
                ax.set_title(title)
                plt.imshow(mpimg.imread(filename), **kwargs)    
            plt.show()
    # check testing dataset content and result
    if True:
        sample_num = 10
        sample_idx = np.random.randint(testing_size, size=sample_num)
        for idx in sample_idx:
            shadow_filename = os.path.join(test_shadow_dir, '%06d.png' % (idx))
            non_shadow_filename = os.path.join(test_non_shadow_dir, '%06d.png' % (idx))
            mask_filename = os.path.join(test_mask_dir, '%06d.png' % (idx))
            Gong_filename = os.path.join(result_Gong_dir, '%06d.png' % (idx))
            Guo_filename = os.path.join(result_Guo_dir, '%06d.png' % (idx))
            Yang_filename = os.path.join(result_Yang_dir, '%06d.png' % (idx))
            STCGAN_filename = os.path.join(result_STCGAN_dir, '%06d.png' % (idx))

            logger.info('Show [ {} ], [ {} ], [ {} ], [ {} ], [ {} ] , [ {} ]   and [ {} ]'.format(shadow_filename, non_shadow_filename, mask_filename, Gong_filename, Guo_filename, Yang_filename, STCGAN_filename))
            
            
            pos_list = [241, 242, 243, 245, 246, 247, 248]
            title_list = ['shadow', 'non_shadow', 'mask', 'Gong', 'Guo', 'Yang', 'ST-CGAN']
            filename_list = [shadow_filename, non_shadow_filename, mask_filename, Gong_filename, Guo_filename, Yang_filename, STCGAN_filename]
            kwargs_list = [{}, {}, {'cmap': 'Greys_r'}, {}, {}, {}, {}]
            
            fig = plt.figure()
            fig.suptitle('Testing data index: {}'.format(idx))
            for pos, title, filename, kwargs in zip(pos_list, title_list, filename_list, kwargs_list):
                ax = fig.add_subplot(pos)
                ax.set_title(title)
                plt.imshow(mpimg.imread(filename), **kwargs)    
            plt.show()

def check_SRD(dst_dir):
    pass

def check_dataset(process_list, dst_root_dir):
    # TODO: check ISTD
    if 'ISTD' in process_list:
        check_ISTD(os.path.join(dst_root_dir, 'ISTD'))    
    # TODO: check SRD
    if 'SRD' in process_list:
        check_SRD(os.path.join(dst_root_dir, 'SRD'))
    return True

def rename_tmp_ISTD(tmp_root_dir):
    logger.info('Start [ rename_tmp_ISTD ]')
    # rename test
    dataset_list = ['test_A', 'test_B', 'test_C']
    for dataset in dataset_list:
        dataset_dir = os.path.join(tmp_root_dir, 'test', dataset)
        logger.info('Rename the directory: [ {} ]'.format(dataset_dir))
        filename_list = os.listdir(dataset_dir)
        for filename in filename_list:
            src_name = os.path.join(dataset_dir, filename)
            
            num1, num2 = os.path.splitext(filename)[0].split('-', 2)
            num1, num2 = int(num1), int(num2)
            new_filename = '%04d%03d.png' % (num1, num2) if num1 >= 100 else '%04d%03d.png' % (100 + num1, num2)
            dst_name = os.path.join(dataset_dir, new_filename)
            os.rename(src_name, dst_name)
            # logger.info('Rename [ {} ] --> [ {} ]'.format(src_name, dst_name))

    # rename result
    result_list = ['Gong', 'Guo', 'Yang', 'ST-CGAN']    
    for result in result_list:
        result_dir = os.path.join(tmp_root_dir, 'result', result)
        logger.info('Rename the directory: [ {} ]'.format(result_dir))
        filename_list = os.listdir(result_dir)
        for filename in filename_list:
            src_name = os.path.join(result_dir, filename)
            num = os.path.splitext(filename)[0].split('_', 2)[0]
            num = int(num)
            new_filename = '%04d.png' % (num) if result != 'Guo' else '%04d.jpg' % (num)
            dst_name = os.path.join(result_dir, new_filename)
            os.rename(src_name, dst_name)
            # logger.info('Rename [ {} ] --> [ {} ]'.format(src_name, dst_name))
    
    
    logger.info('End [ rename_tmp_ISTD ]')
    return

def rename_ISTD(dst_root_dir, src_root_dir):
    # rename train dataset
    logger.info('Start [ rename_ISTD ]')
    
    mode_list = ['train', 'test']
    for mode in mode_list:
        src_shadow_dir = os.path.join(src_root_dir, mode, mode + '_A')
        src_non_shadow_dir = os.path.join(src_root_dir, mode, mode + '_C')
        src_mask_dir = os.path.join(src_root_dir, mode, mode + '_B')
        dst_shadow_dir = os.path.join(dst_root_dir, mode, 'shadow')
        dst_non_shadow_dir = os.path.join(dst_root_dir, mode, 'non_shadow')
        dst_mask_dir = os.path.join(dst_root_dir, mode, 'mask')

        src_list = [src_shadow_dir, src_non_shadow_dir, src_mask_dir]
        dst_list = [dst_shadow_dir, dst_non_shadow_dir, dst_mask_dir]
    
        for src_dir, dst_dir in zip(src_list, dst_list):
            logger.info('Copy [ {} ] --> [ {} ]'.format(src_dir, dst_dir))
            assert(os.path.exists(src_dir))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for idx, filename in enumerate(sorted(os.listdir(src_dir))):
                src_filename = os.path.join(src_dir, filename)
                dst_filename = os.path.join(dst_dir, '%06d.png' % (idx))
                shutil.copy(src_filename, dst_filename)
    # rename experiment result
    result_list = ['Gong', 'Guo', 'Yang', 'ST-CGAN']
    for result_dir in result_list:
        src_dir = os.path.join(src_root_dir, 'result', result_dir)
        dst_dir = os.path.join(dst_root_dir, 'result', result_dir)
        logger.info('Copy [ {} ] --> [ {} ]'.format(src_dir, dst_dir))

        assert(os.path.exists(src_dir))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for idx, filename in enumerate(sorted(os.listdir(src_dir))):
            src_filename = os.path.join(src_dir, filename)
            dst_filename = os.path.join(dst_dir, '%06d.png' % (idx))
            if result_dir == 'Guo':
                img = Image.open(src_filename)
                img.save(dst_filename)
            else:
                shutil.copy(src_filename, dst_filename)

    logger.info('End [ rename_ISTD ]')


def rename_SRD(dst_dir, src_dir):
    # TODO: train dataset(not available)
    # TODO: test dataset
    pass

def resize_ISTD(dst_root_dir):
    logger.info('Start [ resize_ISTD ]')
    # resize testing dataset
    # resize result dataset
    shadow_dir = os.path.join(dst_root_dir, 'test', 'shadow')
    non_shadow_dir = os.path.join(dst_root_dir, 'test', 'non_shadow')
    mask_dir = os.path.join(dst_root_dir, 'test', 'mask')

    Gong_dir = os.path.join(dst_root_dir, 'result', 'Gong')
    Guo_dir = os.path.join(dst_root_dir, 'result', 'Guo')
    Yang_dir = os.path.join(dst_root_dir, 'result', 'Yang')
    STCGAN_dir = os.path.join(dst_root_dir, 'result', 'ST-CGAN')
    result_list = [shadow_dir, non_shadow_dir, mask_dir, Gong_dir, Guo_dir,Yang_dir, STCGAN_dir]
    
    for result in result_list:
        logger.info('Resize [ {} ]'.format(result))
        for filename in os.listdir(result):
            # f = os.path.join(result, filename)
            img = Image.open(os.path.join(result, filename))
            img2 = img.resize((256, 192), Image.BILINEAR)
            img2.save(os.path.join(result, filename))

    logger.info('End [ resize_ISTD ]')
    return

def resize_SRD():
    # TODO:
    return

def rename_tmp_dataset(process_list, tmp_root_dir):
    logger.info('Start [ rename_tmp_dataset ]')
    if 'ISTD' in process_list:
        rename_tmp_ISTD(os.path.join(tmp_root_dir, 'ISTD', 'ISTD_Dataset'))

    logger.info('End [ rename_tmp_dataset ]')
    return

def rename_dataset(process_list, dst_root_dir, src_root_dir):
    logger.info('Start [ rename_dataset ]')
    if 'ISTD' in process_list:
        dst_dir = os.path.join(dst_root_dir, 'ISTD')
        src_dir = os.path.join(src_root_dir, 'ISTD', 'ISTD_Dataset')
        rename_ISTD(dst_dir, src_dir)

    if 'SRD' in process_list:
        dst_dir = os.path.join(dst_root_dir, 'SRD')
        src_dir = os.path.join(src_root_dir, 'SRD')
        rename_SRD(dst_dir, src_dir)

    logger.info('End [ rename_dataset ]')

def resize_dataset(process_list, dst_root_dir):
    logger.info('Start [ resize_dataset ]')
    if 'ISTD' in process_list:
        resize_ISTD(os.path.join(dst_root_dir, 'ISTD'))
    
    if 'SRD' in process_list:
        resize_SRD(os.path.join(dst_root_dir, 'SRD'))
    
    logger.info('End [ resize_dataset ]')
    return

def preprocess_dataset(process_list, dst_root_dir, src_root_dir):
    logger.info('Start [ preprocess_dataset ]')
    tmp_root_dir = 'tmp'
    if not os.path.exists(tmp_root_dir):
        os.makedirs(tmp_root_dir)
    for dataset in process_list:    
        src_dir = os.path.join(src_root_dir, dataset)
        tmp_dir = os.path.join(tmp_root_dir, dataset)
        if not os.path.exists(tmp_dir):
            shutil.copytree(src_dir, tmp_dir)
            logger.info('Create tmp [ {} ]'.format(tmp_dir))
    

    
    rename_tmp_dataset(process_list, tmp_root_dir)
    rename_dataset(process_list, dst_root_dir, tmp_root_dir)
    resize_dataset(process_list, dst_root_dir)
    # TODO: other preprocess option
    
    # remove tmp
    shutil.rmtree(tmp_root_dir, ignore_errors=True)
    logger.info('Remove tmp directory: [ {} ]'.format(tmp_root_dir))
    logger.info('End [ preprocess_dataset ]')

def main(force, process_list, dst_root_dir, src_root_dir):
    
    if os.path.exists(dst_root_dir):
        if force:
            shutil.rmtree(dst_root_dir, ignore_errors=True)
            logger.info('force to delete [ {} ]'.format(dst_root_dir))
            # preprocess datset
            preprocess_dataset(process_list, dst_root_dir, src_root_dir)
        else:
            logger.info('processed dataset is ready')    
    else:
        # preprocess dataset
        preprocess_dataset(process_list, dst_root_dir, src_root_dir)
    
    assert(check_dataset(process_list, dst_root_dir))
    return

if __name__ == '__main__':
    FORCE = False
    process_list = ['ISTD']
    dst_root_dir = 'processed_dataset'
    src_root_dir = 'dataset'
    log_file = os.path.join('log', os.path.basename(__file__) + '.log')
    set_logger(log_file)
    logger = get_logger(__name__)
    main(FORCE, process_list, dst_root_dir, src_root_dir)