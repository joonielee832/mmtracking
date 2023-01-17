from mmdet.datasets import DATASETS
from mmdet.datasets import BDD100KDetDataset

from .coco_video_dataset import CocoVideoDataset


@DATASETS.register_module()
class BDDVideoDataset(CocoVideoDataset):

    CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
               'motorcycle', 'train')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)