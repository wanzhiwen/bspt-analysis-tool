import numpy as np
from shapely.geometry import Polygon

def cal_iou(poly1, poly2):
    '''Calculate the area of the intersection of two polygons
    Args
        poly: 2d array of N x [x1, y1, x2, y2, x3, y3, x4, y4]
    Return
        IOU: 1d array of N iou
    '''
    IOU = np.zeros(len(poly1))
    for i in range(len(poly1)):
        g = Polygon(poly1[i,:8].reshape((4, 2)))
        p = Polygon(poly2[i,:8].reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            IOU[i] = 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            IOU[i] = 0
        else:
            IOU[i] = inter/union
    return IOU

def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou

def overlap_ratio_gt8(gt8_traj, tracker_traj):
    '''Compute overlap ratio between gt8 and rect
    Args
        gt8_traj:2d array N x [x1, y1, x2, y2, x3, y3, x4, y4]
        tracker_traj:2d array of N x [x, y, w, h]
    Return:
        iou
    '''
    tracker_traj_gt8 = np.zeros((len(tracker_traj), 8))
    tracker_traj_gt8[:,0:2] = tracker_traj[:,0:2]
    tracker_traj_gt8[:,2] = tracker_traj[:,0] + tracker_traj[:,2]
    tracker_traj_gt8[:,3] = tracker_traj[:,1]
    tracker_traj_gt8[:,4] = tracker_traj[:,0] + tracker_traj[:,2]
    tracker_traj_gt8[:,5] = tracker_traj[:,1] + tracker_traj[:,3]
    tracker_traj_gt8[:,6] = tracker_traj[:,0]
    tracker_traj_gt8[:,7] = tracker_traj[:,1] + tracker_traj[:,3]
    iou = cal_iou(gt8_traj, tracker_traj_gt8)
    return iou

def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2# check w&h > 0
    if (np.sum(mask) == 0):
        print('no legal gt error')
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        if np.sum(mask) == 0:
            success[i] = 0
        else:
            success[i] = np.sum(iou > thresholds_overlap[i]) / float(np.sum(mask))
            # success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame) # ignore all 0 bb
    return success

def success_overlap_gt8(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    #TODO verify the legality of the data
    mask = np.sum(gt_bb[:, :] == 0, axis=1) != 8 # check all zero rows
    iou = overlap_ratio_gt8(gt_bb[mask], result_bb[mask])# cal iou of gt8
    for i in range(len(thresholds_overlap)):
        #success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)# ignore all 0 bb
        if np.sum(mask) == 0:
            success[i] = 0
            print('no legal gt error')
        else:
            success[i] = np.sum(iou > thresholds_overlap[i]) / float(np.sum(mask))
    return success

def success_error(gt_center, result_center, thresholds, n_frame):
    # n_frame = len(gt_center)
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (51)# -1 -> 51
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    if np.sum(mask) == 0:
        print('no legal gt error')
    for i in range(len(thresholds)):
        # success[i] = np.sum(dist <= thresholds[i]) / float(n_frame) # ignore all 0 bb
        if np.sum(mask) == 0:
            success[i] = 0
        else:
            success[i] = np.sum(dist <= thresholds[i]) / float(np.sum(mask))
    return success