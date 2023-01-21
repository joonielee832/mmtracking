import argparse
import os
import os.path as osp
import random
from tqdm import tqdm
import json

import mmcv
import numpy as np

"""
Each label json file for video file (video_name.json) contains a list of:
- name: str
- frameIndex: int
- videoName: str
- labels: list[dict]
    - id: string
    - category: string
    - attributes:
        - Crowd: boolean
        - Occluded: boolean
        - Truncated: boolean
    - box2d:
        - x1: float
        - y1: float
        - x2: float
        - y2: float

for each frame in videoName

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BDD100k MOT dataset into ReID dataset.')
    parser.add_argument('-i', '--input', help='path of BDD tracking data')
    parser.add_argument('-o', '--output', help='path to save ReID dataset')
    parser.add_argument(
        '--min-per-id',
        type=int,
        default=8,
        help='minimum number of images for each id')
    parser.add_argument(
        '--max-per-id',
        type=int,
        default=1000,
        help='maxmum number of images for each id')
    return parser.parse_args()

def main(args):
    if not osp.isdir(args.output):
        os.makedirs(args.output)
    elif os.listdir(args.output):
        raise OSError(f'Directory must be empty: \'{args.output}\'')
    
    all_reid_imgs_folder = osp.join(args.output, 'images')
    
    # reid_entire_dataset_list = []
    # label_all = 0
    for split in ['train', 'val']:
        # id_cnt = 1  #* remember to str.zfill(8) when saving id
        in_videos_folder = osp.join(args.input, 'images', 'track', split)
        # out_folder = osp.join(args.output, split)
        video_names = os.listdir(in_videos_folder)
        
        #? Crop photos and save
        for video_name in tqdm(sorted(video_names)):
            video_folder = osp.join(in_videos_folder, video_name)
            # raw_img_names = sorted(os.listdir(video_folder))
            gt_json = osp.join(args.input, 'labels', 'box_track_20', split, \
                f"{video_name}.json")
            assert osp.exists(gt_json), f"{gt_json} does not exist!"
            
            gt_json = json.load(open(gt_json))
            # assert len(raw_img_names) == len(gt_json), f"Number of frames don't match up in {video_folder}"

            last_frame_name = ''
            for gt in gt_json:
                # frame_id = gt['frameIndex'] #* id 0 -> 0000001
                frame_name = gt['name']
                assert video_name == gt['videoName'], f"{video_name} video name doesn't match"
                
                if frame_name != last_frame_name:
                    raw_img = mmcv.imread(
                        osp.join(video_folder, frame_name)
                    )
                    last_frame_name = frame_name
                
                for label in gt['labels']:
                    ins_id = label['id']
                    bbox = label['box2d']
                    bbox = np.asarray([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]) #* x1,y1,x2,y2
                    reid_img_folder = osp.join(all_reid_imgs_folder, 
                                               f"{video_name}_{ins_id}")
                    if not osp.exists(reid_img_folder):
                        os.makedirs(reid_img_folder)
                    idx = len(os.listdir(reid_img_folder))
                    reid_img_name = f"{idx:07d}.jpg"
                    reid_img = mmcv.imcrop(raw_img, bbox)
                    mmcv.imwrite(reid_img, osp.join(reid_img_folder, reid_img_name))
                    
                    
        #? Save labels
        reid_meta_folder = osp.join(args.output, 'meta')
        if not osp.exists(reid_meta_folder):
            os.makedirs(reid_meta_folder)
        reid_list = []
        reid_img_folder_names = sorted(os.listdir(all_reid_imgs_folder))
        # num_ids = len(reid_img_folder_names)
        label = 0
        random.seed(0)
        for reid_img_folder_name in reid_img_folder_names:
            reid_img_names = os.listdir(osp.join(all_reid_imgs_folder, reid_img_folder_name))
            #? ignore ids whose number of image is less than min_per_id
            if (len(reid_img_names) < args.min_per_id):
                continue
            #? downsampling when there are too many images owned by one id
            if (len(reid_img_names) > args.max_per_id):
                reid_img_names = random.sample(reid_img_names, args.max_per_id)
            for reid_img_name in reid_img_names:
                reid_list.append(
                    f"{reid_img_folder_name}/{reid_img_name} {label}\n"
                )
            label += 1
            # reid_entire_dataset_list.append(
            #     f"{reid_img_folder_name}/{reid_img_name} {label_all}\n"
            # )
            # label_all += 1
        with open(osp.join(reid_meta_folder, f"{split}.txt"), 'w') as f:
            f.writelines(reid_list)
        
if __name__ == '__main__':
    opts = parse_args()
    main(opts)
    