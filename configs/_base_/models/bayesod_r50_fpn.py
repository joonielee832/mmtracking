_base_ = "./retinanet_r50_fpn.py"

# model settings
num_classes = 80
model = dict(
    detector=dict(
        type="ProbabilisticRetinaNet",
        bbox_head=dict(
            type="ProbabilisticRetinaHead",
            bbox_covariance_type="diagonal",
            loss_cls=dict(
                type='FocalAttenuatedLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
                num_samples=10)
        )
    )
)