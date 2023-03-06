USE_MMDET = True
_base_ = [
    '../../../_base_/models/retinanet_r50_fpn.py',
    '../../../_base_/datasets/mot_challenge_det.py', '../../../_base_/default_runtime.py'
]

mode = dict(
    detector=dict(
        bbox_head=dict(num_classes=1),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/home/misc/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'  
            # https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth
        )
    )
)

#? Configurable settings per experiment
num_gpus = 2
total_epochs = 16    #* originally 4; but 8 cause lr/2 vs Faster R-CNN; 16 bc use full dataset vs half
exp_dir = "retinanet_mot17det_train_exp2"
batch_size = 2  #* set in mot_challenge_det

# data
data_root = '/home/data/MOT17/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(ann_file=data_root + 'annotations/train_cocoformat.json'))
device = 'cuda'

optimizer = dict(type='SGD', lr=0.005*(num_gpus/8)*(batch_size/2), momentum=0.9, weight_decay=0.0001)
evaluation = dict(metric=['bbox'], interval=1, out_dir="/home/results/"+exp_dir)

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
                        'config': {'lr': 0.005*(num_gpus/8)*(batch_size/2), 'batch_size':batch_size*num_gpus},
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
    warmup_ratio=1.0 / 100,
    step=[int(0.75*total_epochs)])