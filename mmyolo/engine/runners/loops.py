import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop

@LOOPS.register_module()
class EpochBasedTrainLoopWith2Loaders(EpochBasedTrainLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 dataloader1: Union[DataLoader, Dict],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1,
                 dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader,
                         max_epochs, val_begin, val_interval, dynamic_intervals)
        
        if isinstance(dataloader1, dict):
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader1 = self.runner.build_dataloader(
                dataloader1, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader1 = dataloader1
            
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        
        dataloader0 = iter(self.dataloader)
        dataloader1 = iter(self.dataloader1)
        
        iters_per_epoch = len(dataloader0)
        
        for idx in range(iters_per_epoch):
            # customized run_iter
            data0_batch = next(dataloader0)
            data1_batch = next(dataloader1)
            
            # call hook before train_iter
            self.runner.call_hook(
                'before_train_iter', batch_idx=idx, data_batch=data0_batch)
            
            # compute detection loss
            losses = self.runner.model.train_step(
                data0_batch, optim_wrapper=self.runner.optim_wrapper)
            
            # compute enhancement loss
            loss_l1 = self.runner.model.train_step(
                data1_batch, optim_wrapper=self.runner.optim_wrapper, is_det=False)
            
            losses.update(loss_l1)

            parsed_losses, log_vars = self.runner.model.parse_losses(losses)  # type: ignore
            self.runner.optim_wrapper.update_params(parsed_losses)
            
            # call hook after train_iter
            self.runner.call_hook(
                'after_train_iter',
                batch_idx=idx,
                data_batch=data0_batch,
                outputs=log_vars)
            
            self._iter += 1
        
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1