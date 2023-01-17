_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/bdd_tracking_det.py', '../../_base_/default_runtime.py'
]

model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/misc/retinanet_r50_fpn_1x_det_bdd100k.pth'  # noqa: E501
        )
    )
)

#? Experiment details
exp_dir = "retinanet_bdd_track_train_exp1"
num_gpus = 1

# optimizer
optimizer = dict(type='SGD', lr=0.02*num_gpus/8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# runtime settings
checkpoint_config = dict(interval=1)
total_epochs = 12
load_from = None
resume_from = None
device = 'cuda' #* needed for training or else runs into error

evaluation = dict(metric=['bbox', 'track'], interval=1, out_dir="/home/results/"+exp_dir)