import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop, BaseLoop

from copy import deepcopy
import numpy as np
from PIL import Image
from tqdm import tqdm

@LOOPS.register_module()
class EpochBasedTrainLoop4EnYOLO(EpochBasedTrainLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 dataloader_det: Union[DataLoader, Dict],
                 dataloader_enh: Union[DataLoader, Dict],
                 max_epochs: int,
                 burnin_epoch: int,
                 val_begin: int = 1,
                 val_interval: int = 1,
                 dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader,
                         max_epochs, val_begin, val_interval, dynamic_intervals)

        self.burnin_epoch = burnin_epoch
        if isinstance(dataloader_enh, dict):
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader_det = self.runner.build_dataloader(
                dataloader_det, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
            self.dataloader_enh = self.runner.build_dataloader(
                dataloader_enh, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader_det = dataloader_det
            self.dataloader_enh = dataloader_enh
            
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()

        if self._epoch < self.burnin_epoch:
            dataloader_det = iter(self.dataloader)  # dataloader for detection before burinin-stage
            dataloader_enh = iter(self.dataloader_enh)  # dataloader for enhancement
    
            iters_per_epoch = len(dataloader_det)
            # do burn-in
            for idx in range(iters_per_epoch):
                # customized run_iter
                data_batch_det = next(dataloader_det)
                data_batch_enh = next(dataloader_enh)

                # call hook before train_iter
                self.runner.call_hook(
                    'before_train_iter', batch_idx=idx, data_batch=data_batch_det)

                # compute detection loss
                losses = self.runner.model.train_step(
                    data_batch_det, optim_wrapper=self.runner.optim_wrapper)

                # compute enhancement loss
                loss_l1 = self.runner.model.train_step(
                    data_batch_enh, optim_wrapper=self.runner.optim_wrapper, is_det=False)

                losses.update(loss_l1)

                parsed_losses, log_vars = self.runner.model.parse_losses(losses)  # type: ignore
                self.runner.optim_wrapper.update_params(parsed_losses)

                # call hook after train_iter
                self.runner.call_hook(
                    'after_train_iter',
                    batch_idx=idx,
                    data_batch=data_batch_det,
                    outputs=log_vars)

                self._iter += 1
        else:
            # do mutual learning
            dataloader_det = iter(self.dataloader)
            dataloader_det_ml = iter(self.dataloader_det)  # dataloader for detection for mutual learning
            dataloader_enh = iter(self.dataloader_enh)  # dataloader for enhancement

            iters_per_epoch = len(dataloader_det)
            
            for idx in range(iters_per_epoch):
                # customized run_iter
                data_batch_det = next(dataloader_det)
                data_batch_det_ml = next(dataloader_det_ml)
                data_batch_enh = next(dataloader_enh)

                # call hook before train_iter
                self.runner.call_hook(
                    'before_train_iter', batch_idx=idx, data_batch=data_batch_det)

                # compute detection loss
                with self.runner.optim_wrapper.optim_context(self.runner):
                    data_det = self.runner.model.data_preprocessor(data_batch_det, True)
                    losses = self.runner.model._run_forward(data_det, mode='loss')
                    # feats_ori = self.runner.model.extract_feat(data_det['inputs'])

                # compute enhancement loss
                loss_l1 = self.runner.model.train_step(
                    data_batch_enh, optim_wrapper=self.runner.optim_wrapper, is_det=False)

                losses.update(loss_l1)

                # enhance real-world underwater images
                data_tmp = self.runner.model.en_data_preprocessor(data_batch_det_ml, training=False)
                data_det_enh = self.runner.model._en_run_forward(data_tmp, mode='tensor')
                
                # compute un-supervised loss
                data_det_enh_tmp = torch.reshape(data_det_enh, list(data_det_enh.shape[:-2])+[-1])
                unsup_loss = torch.mean((torch.mean(data_det_enh_tmp, -1) - 0.5)**2)
                losses.update(dict(unsup_loss=unsup_loss))
                
                # replace labeled underwater detection images with enhanced images
                data_det_ml = self.runner.model.data_preprocessor(data_batch_det_ml, True)
                data_det_ml_ori = deepcopy(data_det_ml)
                data_det_ml_ori['inputs'] = data_det_ml_ori['inputs'].detach()
                data_det_ml['inputs'] = data_det_enh.detach()

                losses_det = {}
                
                losses_tmp_ori = self.runner.model._run_forward(data_det_ml_ori, mode='loss')
                # renew keys
                for k, v in losses_tmp_ori.items():
                    losses_det[k + '_ori'] = v
                losses_tmp_enh = self.runner.model._run_forward(data_det_ml, mode='loss')
                for k, v in losses_tmp_enh.items():
                    losses_det[k + '_enh'] = v
                
                losses.update(losses_det)
                
                feats_ori = self.runner.model.backbone(data_det_ml_ori['inputs'])
                feats_enh = self.runner.model.backbone(data_det_ml['inputs'])

                losses_const = []
                for feat_ori, feat_enh in zip(feats_ori, feats_enh):
                    losses_const.append(nn.L1Loss()(feat_ori, feat_enh.detach()))

                loss_const = dict(loss_const=sum(losses_const))

                losses.update(loss_const)

                parsed_losses, log_vars = self.runner.model.parse_losses(losses)  # type: ignore
                self.runner.optim_wrapper.update_params(parsed_losses)

                # call hook after train_iter
                self.runner.call_hook(
                    'after_train_iter',
                    batch_idx=idx,
                    data_batch=data_batch_det,
                    outputs=log_vars)

                self._iter += 1

        
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
        
        
@LOOPS.register_module()
class EnhanceLoop(BaseLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 outdir: str = None):
        self.outdir = outdir
        super().__init__(runner, dataloader)
        
    def run(self) -> None:
        """Run enhancement."""
        # self.runner.call_hook('before_test')
        # self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        results = []
        for idx, data_batch in tqdm(enumerate(self.dataloader)):
            results.append(self.run_iter(idx, data_batch))
        return results
    
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch."""
        data = self.runner.model.en_data_preprocessor(data_batch, True)
        pred_tensor = self.runner.model._en_run_forward(data, mode='tensor')
        
        out = (torch.clip(pred_tensor, 0, 1) * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        out_img = Image.fromarray(out)
        img_name = data_batch['img_names'][0]
        
        result = dict(image=out_img, img_name=img_name)
        
        return result