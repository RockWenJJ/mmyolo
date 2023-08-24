import math
import os
from copy import deepcopy
from typing import List, Sequence, Tuple, Union, Optional

import mmcv
import mmengine.fileio as fileio
import numpy as np

from mmengine.registry import TRANSFORMS
from mmcv.transforms import LoadImageFromFile, Resize, RandomResize, RandomFlip

@TRANSFORMS.register_module()
class LoadSynImagesFromFile(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        """Funcctions to load both synthetic and clear images"""
        
        img_name = results['img_info']['filename']
        input_prefix = results['input_prefix']
        target_prefix = results['target_prefix']
        
        input_filename = os.path.join(input_prefix, img_name)
        target_filename = os.path.join(target_prefix, img_name)
        
        try:
            if self.file_client_args is not None:
                input_file_client = fileio.FileClient.infer_client(
                    self.file_client_args, input_filename)
                input_img_bytes = input_file_client.get(input_filename)
                
                target_file_client = fileio.FileClient.infer_client(
                    self.file_client_args, target_filename)
                target_img_bytes = target_file_client.get(target_filename)
            else:
                input_img_bytes = fileio.get(
                    input_filename, backend_args=self.backend_args)
                target_img_bytes = fileio.get(
                    target_filename, backend_args=self.backend_args)
            input_img = mmcv.imfrombytes(
                input_img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            target_img = mmcv.imfrombytes(
                target_img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        
        assert input_img is not None, f'failed to load image: {input_filename}'
        assert target_img is not None, f'failed to load image: {target_filename}'
        if self.to_float32:
            input_img = input_img.astype(np.float32)
            target_img = target_img.astype(np.float32)
        
        results['img'] = input_img
        results['target'] = target_img
        results['img_shape'] = input_img.shape[:2]
        results['ori_shape'] = input_img.shape[:2]
        results['file_name'] = img_name
        
        return results
            
        # filename = results[]

@TRANSFORMS.register_module()
class ResizeSynImage(Resize):
    def _resize_target(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get('target', None) is not None:
            if self.keep_ratio:
                target, scale_factor = mmcv.imrescale(
                    results['target'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = target.shape[:2]
                h, w = results['target'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                target, w_scale, h_scale = mmcv.imresize(
                    results['target'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['target'] = target
        
    
    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._resize_target(results)
        return results

@TRANSFORMS.register_module()
class RandomFlipSynImage(RandomFlip):
    def _flip(self, results: dict) -> None:
        super()._flip(results)
        
        # flip target image
        results['target'] = mmcv.imflip(
            results['target'], direction=results['flip_direction'])