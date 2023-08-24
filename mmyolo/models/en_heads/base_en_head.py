from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_plugin_layer, ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.models.backbones.csp_darknet import CSPLayer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS
from ..utils import make_divisible, make_round

@MODELS.register_module()
class BaseEnHead(BaseModule, metaclass=ABCMeta):
    arch_settings = {
        'P5': [[1024, 512, 3, True, False], [512, 256, 9, True, False],
               [256, 128, 6, True, False], [128, 64, 3, True, False]],
        'P6': [[1024, 768, 3, True, False], [768, 512, 3, True, False],
               [512, 256, 9, True, False], [256, 128, 6, True, False],
               [128, 64, 3, True, False]],
               
        # 'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
        #        [256, 512, 9, True, False], [512, 1024, 3, True, True]],
        # 'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
        #        [256, 512, 9, True, False], [512, 768, 3, True, False],
        #        [768, 1024, 3, True, True]]
    }
    def __init__(self,
                 arch_setting: str = 'P5',
                 deepen_factor: float = 0.33,
                 widen_factor: float = 0.5,
                 in_indices: Sequence[int] = (0, 1, 2, 3, 4),
                 out_channels: int = 3,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        
        self.arch_setting = self.arch_settings[arch_setting]
        self.num_stages = len(self.arch_setting)
        
        self.in_indices = in_indices
        self.out_channels = out_channels
        self.widen_factor = widen_factor
        self.deepn_factor = deepen_factor
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        
        self.layers = []
        # build stages
        for idx, setting in enumerate(self.arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            
            self.add_module(f'stage{idx+1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx+1}')
        
        # build output layer
        self.out_layer = self.build_output_layer()
    
    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a enhancement  head stage layer"""
        
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting
        
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepn_factor)
        
        stage = []
        csp_layer = CSPLayer(
            in_channels,
            in_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # different from backbone
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        # add upsample layer
        up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        stage.append(up_layer)
        return stage
    
    def build_output_layer(self) -> nn.Module:
        in_channels = self.arch_setting[-1][1]
        conv = ConvModule(
            make_divisible(in_channels, self.widen_factor),
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        out_layer = nn.Sequential(
            conv,
            up)
        return out_layer
    
    def forward(self, xs):
        xs = list(reversed(xs))
        x = xs[0]
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x) + xs[i+1]
        x = self.out_layer(x)
        return x
    
    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()
    
    
        