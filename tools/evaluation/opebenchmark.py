import numpy as np
from colorama import Style, Fore
from ..utils.statistics import  success_overlap_gt8, success_error

class OPEBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T
    
    def convert_bb_to_center_gt8(self, bboxes):
        return np.array([((bboxes[:, 0] + bboxes[:, 4]) / 2),
                         ((bboxes[:, 1] + bboxes[:, 5]) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)#1e-16 is 

    def eval_success(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names# Default tracker(s)
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]
        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                success_ret_[video.name] = success_overlap_gt8(gt_traj, tracker_traj, n_frame)#
            success_ret[tracker_name] = success_ret_
        return success_ret

    #TODO
    def eval_success_by_label(self, label_number, eval_trackers=None):
        '''
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            dict of result
        '''
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]
        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            gt_label = []
            tracker_label = []
            for video in self.dataset:
                gt_traj = video.gt_traj
                label_traj = video.label
                if tracker_name not in video.pred_trajs:#load if not load
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    #tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = video.pred_trajs[tracker_name]
                n_frame = len(gt_traj)
                if not n_frame == len(tracker_traj) == len(label_traj):
                    print('gt_len = %d,tracker_len = %d, label_len = %d,length not match error!!!'%(n_frame, len(tracker_traj),len(label_traj)))
                    return
                for i in range(n_frame):
                    if label_traj[i][label_number] == 1:
                        gt_label.append(gt_traj[i])
                        tracker_label.append(tracker_traj[i])
            success_ret[tracker_name] = success_overlap_gt8(np.array(gt_label), np.array(tracker_label), len(gt_label))
            return success_ret


    def eval_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            precision_ret: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                gt_center = self.convert_bb_to_center_gt8(gt_traj)#
                tracker_center = self.convert_bb_to_center(tracker_traj)
                thresholds = np.arange(0, 51, 1)
                precision_ret_[video.name] = success_error(gt_center, tracker_center,
                        thresholds, n_frame)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret


    #TODO
    def eval_precision_by_label(self, label_number, eval_trackers=None):
        '''
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            dict of result
        '''
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]
        precision_ret = {}
        for tracker_name in eval_trackers:
            gt_label = []
            tracker_label = []
            for video in self.dataset:
                gt_traj = video.gt_traj
                label_traj = video.label
                if tracker_name not in video.pred_trajs:#load if not load
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    #tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = video.pred_trajs[tracker_name]
                n_frame = len(gt_traj)
                if not n_frame == len(tracker_traj) == len(label_traj):
                    print('gt_len = %d,tracker_len = %d, label_len = %d,length not match error!!!'%(n_frame, len(tracker_traj),len(label_traj)))
                    return
                for i in range(n_frame):
                    if label_traj[i][label_number] == 1:
                        gt_label.append(gt_traj[i])
                        tracker_label.append(tracker_traj[i])
            
            gt_center = self.convert_bb_to_center_gt8(np.array(gt_label))#
            tracker_center = self.convert_bb_to_center(np.array(tracker_label))
            thresholds = np.arange(0, 51, 1)
            precision_ret[tracker_name] = success_error(gt_center, tracker_center, thresholds, len(gt_center))
            return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path, 
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[video.name] = success_error(gt_center_norm,
                        tracker_center_norm, thresholds, n_frame)
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

