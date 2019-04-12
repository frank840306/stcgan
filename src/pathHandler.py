import os
import datetime

class PathHandler():
    def __init__(self, root_dir='.', task='stcgan', data='ISTD'):
        now = datetime.datetime.now()
        self.root_dir = root_dir
        
        # set dataset directory
        self.data_dir = os.path.join(self.root_dir, '..', 'processed_dataset', data)
        self.set_data_dir()

        # set task directory
        self.task_dir = os.path.join(self.root_dir, 'task', task, data)

        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)
        self.set_task_dir()

    def set_data_dir(self):
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.train_mask_dir = os.path.join(self.train_dir, 'mask')
        self.train_shadow_dir = os.path.join(self.train_dir, 'shadow')
        self.train_shadow_free_dir = os.path.join(self.train_dir, 'non_shadow')
        
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.test_mask_dir = os.path.join(self.test_dir, 'mask')
        self.test_shadow_dir = os.path.join(self.test_dir, 'shadow')
        self.test_shadow_free_dir = os.path.join(self.test_dir, 'non_shadow')
        
    def set_task_dir(self):
        self.mdl_dir = os.path.join(self.task_dir, 'mdl')
        self.log_dir = os.path.join(self.task_dir, 'log')
        
        self.valid_dir = os.path.join(self.task_dir, 'valid')
        self.valid_mask_dir = os.path.join(self.valid_dir, 'mask')
        self.valid_shadow_free_dir = os.path.join(self.valid_dir, 'non_shadow')
        self.valid_gt_mask_dir = os.path.join(self.valid_dir, 'gt_mask')
        self.valid_gt_shadow_dir = os.path.join(self.valid_dir, 'gt_shadow')
        self.valid_gt_shadow_free_dir = os.path.join(self.valid_dir, 'gt_non_shadow')

        self.result_dir = os.path.join(self.task_dir, 'result')
        self.result_shadow_dir = os.path.join(self.result_dir, 'shadow')
        self.result_shadow_free_dir = os.path.join(self.result_dir, 'non_shadow')
        self.result_mask_dir = os.path.join(self.result_dir, 'mask')
        self.result_resnet_dir = os.path.join(self.result_dir, 'resnet_non_shadow')
        
        dir_list = [self.mdl_dir, self.log_dir, self.valid_dir, self.valid_mask_dir, self.valid_shadow_free_dir, self.valid_gt_mask_dir, self.valid_gt_shadow_dir, self.valid_gt_shadow_free_dir, self.result_dir, self.result_shadow_dir, self.result_shadow_free_dir, self.result_mask_dir, self.result_resnet_dir] 
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)
        
    