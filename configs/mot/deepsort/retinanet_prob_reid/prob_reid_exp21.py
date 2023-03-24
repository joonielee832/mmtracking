_base_ = [
    '../../../_base_/models/retinanet_r50_fpn.py',
    '../../../_base_/datasets/mot_challenge.py', '../../../_base_/default_runtime.py'
]
model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(
            bbox_coder=dict(clip_border=False), 
            num_classes=1,
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/results/retinanet_mot17det_train_exp7/latest.pth'   #* configurable
            # '/home/misc/retinanet_epoch_4.pth'
        )
    ),
    motion=dict(type='KalmanFilter', center_only=False),
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
            num_samples=100,
            loss_attenuation=dict(attenuated=True, alpha=0.1, epoch_step=None, iters_in_epoch=None),
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0, prob=False, num_samples=100),
            loss_uncertainty=dict(
                type='FeatureUncertaintyLoss', margin_exp=1, loss_weight=0.001),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
                '/home/results/prob_reid_mot17_train_exp21/latest.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SortTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=0.3,    #* originally 2.0 (classic); double if 'Frobenius'; configurable
            prob=dict(mode='KL', num_samples=100)),  #* mode ('Jensen', 'Frobenius', 'KL'); if 'Jensen', set num_samples
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        alpha=0.0,
        num_frames_retain=100))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
