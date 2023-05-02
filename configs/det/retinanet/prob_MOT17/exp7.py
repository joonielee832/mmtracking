USE_MMDET = True
_base_ = ['./exp1.py']

num_gpus = 4
total_epochs = 4
step = 3
exp_dir = "bayesod_mot17det_train_exp7"
iters_in_epoch = 3996
lr_factor = 0.25

model = dict(
    detector=dict(
        bbox_head=dict(
            bbox_coder=dict(clip_border=False), 
            num_classes=1,
            use_pos_mask=False,
            with_nms=True,
            epoch_step=step-1,
            iters_in_epoch=iters_in_epoch/(num_gpus),
            loss_cls=dict(attenuated=True),
            loss_bbox=dict(type='SmoothL1WithNLL', attenuated=True),
            init_cfg=
                dict(type='Xavier', layer='Conv2d', override=[
                    dict(type='Normal', name='retina_cls_var', layer='Conv2d', bias=-10.0, std=0.01),
                    dict(type='Normal', name='retina_reg_cov', layer='Conv2d', bias=0.0, std=0.0001)
                ]))
    )
)

optimizer = dict(type='SGD', lr=0.01*lr_factor*(num_gpus/8), momentum=0.9, weight_decay=0.0001)
evaluation = dict(metric=['bbox'], interval=1, out_dir="/home/results/"+exp_dir, save_best="bbox_mAP")
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',
             log_dir="/home/results/"+exp_dir+"/tf_board",
             interval=50),
        dict(type='WandbLoggerHook',
            init_kwargs={'name': exp_dir,
                        'project': 'retinanet_mot17_det',
                         'dir': "/home/results/"+exp_dir,
                         'sync_tensorboard': True,
                        'config': {'lr': 0.01*(num_gpus/8), 'batch_size':2*num_gpus},
                        'notes': '',
                        'resume': 'allow',   # set to must if need to resume; set id corresponding to run
                        # 'id': 'nnknkq8u'
                        },
            interval=50)
    ])

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01*lr_factor*(num_gpus/8),
    step=[step])