USE_MMDET = True
_base_ = [
    '../../../_base_/models/bayesod_r50_fpn.py',
    '../../../_base_/datasets/mot_challenge_det.py', '../../../_base_/default_runtime.py'
]

model = dict(
    detector=dict(
        bbox_head=dict(
            bbox_coder=dict(clip_border=False), 
            num_classes=1,
            use_pos_mask=True,
            with_nms=True,
            loss_bbox=dict(annealing_step=int(10000)),
            init_cfg=
                dict(type='Xavier', layer='Conv2d', override=[
                    dict(type='Xavier', name='retina_cls_var', layer='Conv2d', bias=-10.0),
                    dict(type='Xavier', name='retina_reg_cov', layer='Conv2d', bias=0.0)
                ]))
    )
)

#? Configurable settings per experiment
num_gpus = 1
total_epochs = 4
exp_dir = "bayesod_mot17det_train_exp1"

device = 'cuda'

optimizer = dict(type='SGD', lr=0.01*(num_gpus/8), momentum=0.9, weight_decay=0.0001)
evaluation = dict(metric=['bbox'], interval=1, out_dir="/home/results/"+exp_dir, save_best="bbox_mAP")

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',
             log_dir="/home/results/"+exp_dir+"/tf_board",
             interval=50),
        # dict(type='WandbLoggerHook',
        #     init_kwargs={'name': exp_dir,
        #                 'project': 'retinanet_mot17_det',
        #                  'dir': "/home/results/"+exp_dir,
        #                  'sync_tensorboard': True,
        #                 'config': {'lr': 0.01*(num_gpus/8), 'batch_size':2*num_gpus},
        #                 'notes': '',
        #                 'resume': 'allow',   # set to must if need to resume; set id corresponding to run
        #                 # 'id': 'nnknkq8u'
        #                 },
        #     interval=50)
    ])

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01*(num_gpus/8),
    step=[3])