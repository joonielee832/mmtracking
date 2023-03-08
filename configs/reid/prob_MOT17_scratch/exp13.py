_base_ = [
    '../../_base_/datasets/mot_challenge_reid.py', '../../_base_/default_runtime.py'
]

TRAIN_REID = True

#? Experiment details
exp_dir = "prob_reid_mot17_train_exp13"
num_gpus = 2
total_epochs = 12    #* originally 6
load_from = None
resume_from = None
batch_size = 1

model = dict(
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='ProbabilisticReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            lce_sample_weight=0.1,  #* weight for cross entropy sample loss; configurable
            num_samples=10,         #* num samples for cross entropy; configurable
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0, prob=True, num_samples=100),  #* num_samples configurable
            loss_uncertainty=dict(
                type='FeatureUncertaintyLoss', margin_exp=6, loss_weight=0.001),    #* margin_exp and loss_weight configurable
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
                '/home/misc/resnet50_batch256_imagenet_20200708-cfb998bf.pth'  # noqa: E501
        )))
# optimizer
optimizer = dict(type='SGD', lr=0.1*num_gpus/8 * batch_size/1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[total_epochs-1])

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',
             log_dir="/home/results/"+exp_dir+"/tf_board",
             interval=50),
        # dict(type='WandbLoggerHook',
        #     init_kwargs={'name': exp_dir,
        #                 'project': 'resnet_prob_reid_mot17',
        #                  'dir': "/home/results/"+exp_dir,
        #                  'sync_tensorboard': True,
        #                 'config': {'lr': 0.1*num_gpus/8*batch_size/1, 'batch_size':batch_size*num_gpus},
        #                 'notes': '',
        #                 'resume': 'allow',   # set to must if need to resume; set id corresponding to run
        #                 'mode': 'offline',
        #                 'id': 'sv56jwmw'
        #                 },
        #     interval=50)
    ]
)

device = 'cuda'
