import os
import json
from .dataset import Dataset
from .video import Video
class SoccerVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        absent: player disappear
        attribute: video attribute
        label: frame attribute
    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect, absent, attribute, label, load_img=False):
        super(SoccerVideo, self).__init__(name, root, video_dir, init_rect, img_names, gt_rect, absent, attribute, label, load_img)

class SoccerDataset(Dataset):
    def __init__(self, name, dataset_root, load_img=False):
        super(SoccerDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)
        # load videos
        videos = meta_data.keys()
        self.videos = {}
        for video in videos:
            self.videos[video] = SoccerVideo(video, 
                                        dataset_root, 
                                        meta_data[video]['video_dir'], 
                                        meta_data[video]['init_rect'], 
                                        meta_data[video]['img_names'], 
                                        meta_data[video]['gt_rect'],
                                        meta_data[video]['absent'],
                                        meta_data[video]['attribute'],
                                        meta_data[video]['label'],
                                        load_img)
