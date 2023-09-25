_base_ = './enyolov5_s-v61_fast_1xb16-400e_duo.py'

model = dict(
    en_head=dict(
        act_cfg=dict(type='ReLU', inplace=True)))
