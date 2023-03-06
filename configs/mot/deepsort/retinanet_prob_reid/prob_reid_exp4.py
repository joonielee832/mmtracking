_base_ = [
    '../../../_base_/models/retinanet_r50_fpn.py',
    '../../../_base_/datasets/mot_challenge.py', '../../../_base_/default_runtime.py'
]
model = dict(
    type='DeepSORT',
    detector=dict(
        bbox_head=dict(num_classes=1),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/results/retinanet_mot17det_train_exp1/latest.pth'  
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
            lce_sample_weight=0.1,
            num_samples=10,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0, prob=True, num_samples=100),
            loss_uncertainty=dict(
                type='FeatureUncertaintyLoss', margin_exp=1, loss_weight=0.001),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/results/prob_reid/prob_reid_mot17_train_exp4/latest.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SortTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
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
