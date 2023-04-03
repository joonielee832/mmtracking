USE_MMDET = True
_base_ = ['./exp2.py']
# data
data_root = '/home/data/MOT17/'
data = dict(
    train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
    test=dict(ann_file=data_root + 'annotations/train_cocoformat.json'))
device = 'cuda'

num_gpus = 2
exp_dir = "bayesod_mot17det_train_exp4"

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