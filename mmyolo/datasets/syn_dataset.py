import json
import os
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
from glob import glob

# from mmdet.datasets import Compose
from ..registry import DATASETS
from mmengine.dataset import Compose
# from .pipelines import Compose


@DATASETS.register_module()
class SynDataset(Dataset):
    # CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')  # not useful for uie
    
    def __init__(self,
                 pipeline,
                 ann_file,
                 data_root=None,
                 data_prefix=dict(input='synthesis/', target='images/'),
                 img_suffix='.png',
                 test_mode=False):
        super().__init__()
        self.data_root = data_root
        self.input_prefix = os.path.join(data_root, data_prefix['input'])
        self.target_prefix = os.path.join(data_root, data_prefix['target'])
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        self.ann_file = os.path.join(data_root, ann_file)
        
        self.data_infos = self.load_annotations(self.ann_file)
        self._set_group_flag()
        self.pipeline = Compose(pipeline)
    
    def __getitem__(self, idx):
        '''Get training/test data from pipeline.
        Args:
            idx (int): Index of data
        Returns:
            data (dict): Training/test data
        '''
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        '''Prepare results dict for pipeline.'''
        results['input_prefix'] = self.input_prefix
        results['target_prefix'] = self.target_prefix
        return results
    
    def load_annotations(self, ann_file):
        '''Load annotation from annotation file.'''
        # return mmcv.load(ann_file)
        return json.load(open(ann_file, 'r'))
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def __len__(self):
        return len(self.data_infos)
    
    def get_ann_info(self, idx):
        return self.data_info[idx]['ann']