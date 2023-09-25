# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower
from mmyolo.engine import Runner4EnYOLO

from glob import glob
from PIL import Image
import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA


# TODO: support fuse_conv_bn
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--json-prefix',
        type=str,
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.deploy:
        cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

    # add `format_only` and `outfile_prefix` into cfg
    if args.json_prefix is not None:
        cfg_json = {
            'test_evaluator.format_only': True,
            'test_evaluator.outfile_prefix': args.json_prefix
        }
        cfg.merge_from_dict(cfg_json)

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    if args.tta:
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
                                   " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
                                      "in config. Can't use tta !"

        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        # runner = Runner.from_cfg(cfg)
        try:
            runner = Runner4EnYOLO.from_cfg(cfg)
        except:
            runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    # runner.test()
    
    colors = ['green', 'blue', 'orange']
    img_dirs = [
        "/mnt/03_Data/DUO/DUO_resized/images/test_various_envs/test_greenish",
        "/mnt/03_Data/DUO/DUO_resized/images/test_various_envs/test_bluish",
        "/mnt/03_Data/DUO/DUO_resized/images/test_various_envs/test_turbid"
    ]
    n_pt = 80
    # img_dirs = [
    #     "/mnt/03_Data/DUO/DUO_various_envs/test_no_robo/test_bright_green",
    #     "/mnt/03_Data/DUO/DUO_various_envs/test_no_robo/test_blue",
    #     "/mnt/03_Data/DUO/DUO_various_envs/test_no_robo/test_dark",
    # ]
    
    labels = ["Green", "Blue", "Turbid"]
    
    feats = []
    
    for i, img_dir in enumerate(img_dirs):
        imgs = glob(os.path.join(img_dir, "*.png"))
        # imgs = glob(os.path.join(img_dir, "*.jpg"))
        imgs.sort()
        
        features = []
        # for img in imgs[:n_pt]:
        for img in imgs[(i+1)*n_pt:(i+2)*n_pt]:
            im = Image.open(img)
            im = im.resize((640, 360))
            
            img_data = np.array(im)[:, :, ::-1]
            img_data = np.expand_dims(img_data, axis=0)
            img_data = torch.Tensor(img_data.copy())
            img_data = torch.permute(img_data, (0, 3, 1, 2)).to('cuda')
            img_data = img_data / 255.
            
            model_feats = runner.model.backbone(img_data)
            
            features.append(np.reshape(model_feats[-1].detach().cpu().numpy(), (1, -1)))
        
        feat = np.concatenate(features, 0)
        feats.append(feat)
        
    pca = KernelPCA(n_components=2, kernel='cosine') #poly/cosine/sigmoid/rbf
    
    X_tsnes = []
    for i, feat in enumerate(feats):
        pca_data = pca.fit_transform(feat)

        # if i == 0:
        #     pca_data = pca_data * 0.3
        #     pca_data[:, 0] = pca_data[:, 0] + 0.1
        # elif i == 1:
        #     pca_data = pca_data * 0.3
        #     pca_data[:, 0] = pca_data[:, 0] - 0.1
        # else:
        #     pca_data = pca_data * 0.4
        #     pca_data[:, 0] = pca_data[:, 0] + 0.1
        #     pca_data[:, 1] = pca_data[:, 1] - 0.04
        # pca_data = pca_data * 0.1
        # if i == 0:
        #     pca_data = pca_data * 0.4
        # elif i == 1:
        #     pca_data = pca_data * 0.35
        # else:
        #     pca_data = pca_data * 0.45
        
        
        X_tsnes.append(pca_data)

    from matplotlib import gridspec
    from mpl_toolkits.mplot3d import Axes3D

    plt.rc('font', family="Times New Roman")
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1, height_ratios=[1])
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax0 = plt.subplot(gs[0])
    # ax = Axes3D(fig)

    for i, X_tsne in enumerate(X_tsnes):
        # if i == 3:
        #     X_tsne[X_tsne[:, 0] > 0.1] = None
    
        ax0.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100, marker='o', color=colors[i], edgecolor='none', label=labels[i],
                    alpha=0.9)
        # ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],s=100, marker='o', color=colors[i], edgecolor='none', label=labels[i],
        #             alpha=0.9)
        # print(np.mean(X_tsne, 0))

    plt.xticks([])
    plt.yticks([])
    # plt.xlim(-0.3, 0.3)
    # plt.ylim(-0.2, 0.2)
    # plt.axis('equal')
    # plt.legend(loc=1)
    # plt.savefig("tSNE_before.pdf")
    plt.show()
            


if __name__ == '__main__':
    main()
