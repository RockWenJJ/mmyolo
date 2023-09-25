import json
import os
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import cv2

# from mmdet.datasets import Compose
from ..registry import DATASETS
from mmengine.dataset import Compose
# from .pipelines import Compose


@DATASETS.register_module()
class RwDataset(Dataset):
    def __init__(self,
                 pipeline,
                 data_root=None,
                 img_suffix='.png'):
        super().__init__()
        self.data_root = data_root
        self.img_suffix = img_suffix
        self.images = glob(os.path.join(data_root, "*"+img_suffix))
        
        self.data_infos = self._get_data_infos()
        self._set_group_flag()
        self.pipeline = Compose(pipeline)
    
    def __getitem__(self, idx):
        '''Get training/test data from pipeline.
        Args:
            idx (int): Index of data
        Returns:
            data (dict): Training/test data
        '''
        # RwDataset is only used for real-world test
        return self.prepare_test_img(idx)
        # if self.test_mode:
        #     return self.prepare_test_img(idx)
        # while True:
        #     data = self.prepare_train_img(idx)
        #     if data is None:
        #         idx = self._rand_another(idx)
        #         continue
        #     return data
    
    # def prepare_train_img(self, idx):
    #     img_info = self.data_infos[idx]
    #     results = dict(img_info=img_info)
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)
    
    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        '''Prepare results dict for pipeline.'''
        results['input_prefix'] = self.data_root
        return results
    
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
    
    def _get_data_infos(self):
        infos = []
        for image in self.images:
            info=dict(filename=os.path.basename(image))
            img = cv2.imread(image)
            h, w = img.shape[1:]
            info['width'], info['height'] = h, w
            infos.append(info)
        return infos
            
    def __len__(self):
        return len(self.data_infos)
    
    # def get_ann_info(self, idx):
    #     return self.data_info[idx]['ann']