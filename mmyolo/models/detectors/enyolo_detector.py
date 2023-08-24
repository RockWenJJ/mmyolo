import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmyolo.registry import MODELS
from mmengine.optim import OptimWrapper

@MODELS.register_module()
class EnYOLODetector(SingleStageDetector):
    r"""Implementation of YOLO Series with Image Enhancement
        """
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 en_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 en_data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.en_head = MODELS.build(en_head)
        self.en_data_preprocessor = MODELS.build(en_data_preprocessor)
        
        
        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')
            
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,
                   is_det: bool = True) -> Dict[str, torch.Tensor]:
        if is_det:
            # super().train_step(data, optim_wrapper)
            with optim_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                losses = self._run_forward(data, mode='loss')
        else:
            with optim_wrapper.optim_context(self):
                data = self.en_data_preprocessor(data, True)
                losses = self._en_run_forward(data, mode='loss') # mode should be 'loss' or 'tensor'
        
        return losses
    
    def _en_run_forward(self, data: Union[dict, tuple, list],
                        mode: str) -> Union[Dict[str, torch.Tensor], list]:
        if isinstance(data, dict):
            results = self.run_enhancement(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self.run_enhancement(*data, mode=mode)
        else:
            raise TypeError('Output of `en_data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        '''Replace the original extract feat because the backbone feats are
        different are longer than the neck feats.'''
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x[2:])
        return x
    
    def run_enhancement(self,
                        inputs: torch.Tensor,
                        targets: torch.Tensor,
                        img_names: Optional[list] = None,
                        data_samples: Optional[list] = None,
                        mode: str = 'loss') -> Union[Dict[str, torch.Tensor], list]:
        # extract backbone feats
        backbone_feats = self.backbone(inputs)
        x_res = self.en_head(backbone_feats)
        
        pred = x_res + inputs
        
        if mode == 'loss':
            loss_l1 = nn.L1Loss()(pred, targets)
            return {'loss_l1':loss_l1}
        elif mode == 'tensor':
            return pred
        else:
            raise NotImplementedError(f'{mode} is invalid when run for enhancement.')
        
        
        