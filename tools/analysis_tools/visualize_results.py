# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from multiprocessing import Pool

import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import ProgressBar, check_file_exist, mkdir_or_exist

from mmdet.datasets import get_loading_pipeline
from mmdet.evaluation import eval_map
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmdet.utils import replace_cfg_vals, update_data_root
from mmdet.visualization import DetLocalVisualizer

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower
from mmyolo.engine import Runner4EnYOLO



class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True.
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0.
        runner (:obj:`Runner`): The runner of the visualization process.
    """

    def __init__(self, show=False, wait_time=0, score_thr=0, runner=None):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr
        self.visualizer = DetLocalVisualizer()
        self.runner = runner
        self.evaluator = runner.test_evaluator

    def _save_image_gts_results(self,
                                dataset,
                                results,
                                out_dir=None,
                                task='det'):
        """Display or save image with groung truths and predictions from a
        model.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection results from test results pkl file.
            out_dir (str, optional): The filename to write the image.
                Defaults: None.
            task (str): The task to be performed. Defaults: 'det'
        """
        mkdir_or_exist(out_dir)

        prog_bar = ProgressBar(len(dataset))
        
        for index, d in enumerate(dataset):
            data_info = dataset[index]
            data_info['gt_instances'] = data_info['instances']

            # calc save file path
            filename = data_info['img_path']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname +  name
            out_file = osp.join(out_dir, save_filename)

            if task == 'det':
                gt_instances = InstanceData()
                gt_instances.bboxes = results[index]['gt_instances']['bboxes']
                gt_instances.labels = results[index]['gt_instances']['labels']

                pred_instances = InstanceData()
                pred_instances.bboxes = results[index]['pred_instances'][
                    'bboxes']
                pred_instances.labels = results[index]['pred_instances'][
                    'labels']
                pred_instances.scores = results[index]['pred_instances'][
                    'scores']

                data_samples = DetDataSample()
                data_samples.pred_instances = pred_instances
                data_samples.gt_instances = gt_instances

            img = mmcv.imread(filename, channel_order='rgb')
            self.visualizer.add_datasample(
                'image',
                img,
                data_samples,
                show=self.show,
                draw_gt=False,
                pred_score_thr=self.score_thr,
                out_file=out_file)
            prog_bar.update()

    def show_result(self,
             dataset,
             results,
             show_dir='work_dir'):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection results from test results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
        """

        self.visualizer.dataset_meta = dataset.metainfo
        
        out_dir = osp.join(osp.abspath(show_dir), 'results_vis')
        self._save_image_gts_results(dataset, results, out_dir)
        


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    cfg.test_dataloader.dataset.test_mode = True

    cfg.test_dataloader.pop('batch_size', 0)
    if cfg.train_dataloader.dataset.type in ('MultiImageMixDataset',
                                             'ClassBalancedDataset',
                                             'RepeatDataset', 'ConcatDataset'):
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.dataset.pipeline)
    else:
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.pipeline)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    outputs = load(args.prediction_path)

    cfg.work_dir = args.show_dir
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        # runner = Runner.from_cfg(cfg)
        runner = Runner4EnYOLO.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    result_visualizer = ResultVisualizer(args.show, args.wait_time,
                                         args.show_score_thr, runner)
    result_visualizer.show_result(
        dataset, outputs, show_dir=args.show_dir)


if __name__ == '__main__':
    main()
