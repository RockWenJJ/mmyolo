import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop

@LOOPS.register_module()
class EpochBasedTrainLoop4EnYOLO(EpochBasedTrainLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
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
            self.dataloader_enh = self.runner.build_dataloader(
                dataloader_enh, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader_enh = dataloader_enh
            
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        
        dataloader_det = iter(self.dataloader)     # dataloader for detection
        dataloader_enh = iter(self.dataloader_enh)    # dataloader for enhancement
        
        iters_per_epoch = len(dataloader_det)

        if self._epoch < self.burnin_epoch:
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
            for idx in range(iters_per_epoch):
                # customized run_iter
                data_batch_det = next(dataloader_det)
                data_batch_enh = next(dataloader_enh)

                # call hook before train_iter
                self.runner.call_hook(
                    'before_train_iter', batch_idx=idx, data_batch=data_batch_det)

                # compute detection loss
                with self.runner.optim_wrapper.optim_context(self.runner):
                    data_det = self.runner.model.data_preprocessor(data_batch_det, True)
                    losses = self.runner.model._run_forward(data_det, mode='loss')
                    feats_ori = self.runner.model.extract_feat(data_det['inputs'])

                # compute enhancement loss
                loss_l1 = self.runner.model.train_step(
                    data_batch_enh, optim_wrapper=self.runner.optim_wrapper, is_det=False)

                losses.update(loss_l1)

                # enhance real-world underwater images
                data_tmp = self.runner.model.en_data_preprocessor(data_batch_det, training=False)
                data_det_enh = self.runner.model._en_run_forward(data_tmp, mode='tensor')

                # replace labeled underwater detection images with enhanced images
                data_det['inputs'] = torch.clip(data_det_enh, 0., 1.)

                losses_tmp = self.runner.model._run_forward(data_det, mode='loss')
                feats_enh = self.runner.model.extract_feat(data_det['inputs'])

                # compute adversarial loss
                losses_det = {}
                # renew keys
                for k, v in losses_tmp.items():
                    losses_det[k+'1'] = v

                losses.update(losses_det)

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