import bisect
import logging
import time
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader


from mmengine.registry import RUNNERS, LOOPS
from mmengine.runner import Runner as BaseRunner
from mmengine.runner import BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop

@RUNNERS.register_module()
class Runner4EnYOLO(BaseRunner):
    def __init__(self, *args, **kwargs):
        train_dataloader_det = kwargs.pop('train_dataloader_det')
        train_dataloader_enh = kwargs.pop('train_dataloader_enh')
        self._train_dataloader_det = train_dataloader_det
        self._train_dataloader_enh = train_dataloader_enh
        
        self._val_enh_loop = kwargs.pop('val_enh_cfg')
        self._val_dataloader_enh = kwargs.pop('val_dataloader_enh')
        
        super().__init__(*args, **kwargs)
        
        
    @classmethod
    def from_cfg(cls, cfg):
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            train_dataloader_det=cfg.get('train_dataloader_det'),
            train_dataloader_enh=cfg.get('train_dataloader_enh'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            val_dataloader_enh=cfg.get('val_dataloader_enh'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            val_enh_cfg=cfg.get('val_enh_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )
        return runner
    
    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')
    
        loop_cfg = copy.deepcopy(loop)
    
        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')
    
        if 'type' in loop_cfg:
            if loop_cfg['type'] == 'EpochBasedTrainLoop4EnYOLO':
                loop = LOOPS.build(
                    loop_cfg,
                    default_args=dict(
                        runner=self, dataloader=self._train_dataloader,
                        dataloader_det=self._train_dataloader_det,
                        dataloader_enh=self._train_dataloader_enh))
            else:
                loop = LOOPS.build(
                    loop_cfg,
                    default_args=dict(
                        runner=self, dataloader=self._train_dataloader,
                    ))
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
        return loop  # type: ignore
    
    def enhance(self):
        # 0. build enhance_loop, where the enhance dataloader will be built.
        self._enhance_loop = self.build_enhance_loop(self._val_enh_loop)
        
        self.call_hook('before_run')
        
        self.load_or_resume()
        
        results = self.enhance_loop.run()
        
        return results
        
    def build_enhance_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'enhance_loop should be a Loop object or dict, but got {loop}')
        
        loop_cfg = copy.deepcopy(loop)
        
        if 'type' in loop_cfg:
            loop = LOOPS.build(
                    loop_cfg,
                    default_args=dict(
                        runner=self,
                        dataloader=self._val_dataloader_enh)
            )
        else:
            raise TypeError(f'The type of enhance_cfg is unknown')
        
        return loop
    
    @property
    def enhance_loop(self):
        if isinstance(self._enhance_loop, BaseLoop) or self._enhance_loop is None:
            return self._enhance_loop
        else:
            self._enhance_loop = self.build_enhance_loop(self._enhance_loop)
            return self._enhance_loop
        
        
        