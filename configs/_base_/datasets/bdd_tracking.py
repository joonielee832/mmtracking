# dataset settings
dataset_type = 'BDDVideoDataset'
data_root = '/home/data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root +
            'jsons/box_track_train_cocofmt.json',
            img_prefix=data_root + 'images/track/train/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=3,
            filter_key_img=True,
            method='uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + 'jsons/det_train_cocofmt.json',
            img_prefix=data_root + 'images/100k/train/',
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'jsons/box_track_val_cocofmt.json',
        img_prefix=data_root + 'images/track/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'jsons/box_track_val_cocofmt.json',
        img_prefix=data_root + 'images/track/val/',
        pipeline=test_pipeline))