import argparse
from glob import glob
import os
import numpy as np
#from sklearn.metrics import auc
from tools.database.soccer import SoccerDataset
from tools.evaluation.opebenchmark import OPEBenchmark
from tools.visualization.draw_precision_success import draw_success_precision

parser = argparse.ArgumentParser(description='eval tracking result')
parser.add_argument('--tracker_path', '-p', type=str, help='tracker result path')
parser.add_argument('-database' , '-d', type=str, help='database path')
parser.add_argument('--tracker_prefix', '-t', default='',type=str, help='tracker name')
parser.set_defaults(database='SoccerDatabase')
args = parser.parse_args()

def main():
    trackers = glob(os.path.join(args.tracker_path, args.tracker_prefix+'*'))
    trackers = [os.path.basename(x) for x in trackers]
    assert len(trackers) > 0
    dataset = SoccerDataset('Soccer', args.database)
    dataset.set_tracker(args.tracker_path, trackers)
    benchmark = OPEBenchmark(dataset)



    #dataset.videos['shot121'].load_tracker(dataset.tracker_path, 'BACF', True)
    #print(dataset.videos['shot121'].pred_trajs['BACF'])
    #print(dataset.videos['shot121'].gt_traj)
    #precision_ret = benchmark.eval_precision()
    #print(benchmark.eval_success_by_label(label_number=0))
    #print(benchmark.eval_precision_by_label(label_number=7))
    #success_ret = benchmark.eval_success()
    # print(success_ret[args.tracker_prefix])
    # precision_ret = benchmark.eval_precision()
    # print(precision_ret[args.tracker_prefix])
    
    # #eval by label
    # success = {}
    # trackers = dataset.tracker_names
    # for tracker in trackers:
    #     t = []
    #     for i in range(8):
    #         a = benchmark.eval_success_by_label(label_number=i)
    #         t.append(a[tracker])
    #     t = np.array(t)
    #     success[tracker] = t
    #     np.savetxt('./data.txt',t,delimiter=' ',newline='\n',fmt = '%.3f')


    # eval by label2 (precision)
    # precision_ret = benchmark.eval_precision()
    # trackers = dataset.tracker_names
    # success = {}
    # for tracker in trackers:
    #     print('tracker:' + tracker)
    #     videos = precision_ret[tracker]
    #     t = []
    #     auc_cal = []
    #     for i in range(8):#8 labels
    #         count = 0
    #         results = {}
    #         for video_name in videos:
    #             if dataset[video_name].attribute[i] == 1:
    #                 results[video_name] = precision_ret[tracker][video_name]
    #                 count = count+1
    #         value = [v for k, v in results.items()]
    #         result = np.mean(value, axis = 0)[20]
    #         t.append(result)
    #         print('attribute = ' + str(i) + ', num = ' + str(count))
    #     success[tracker] = t
    #     np.savetxt('./data.txt',t,delimiter=' ',newline='\n',fmt = '%.3f')# 只取20像素那列

    # eval by label2 (success)
    # success_ret = benchmark.eval_success()
    # trackers = dataset.tracker_names
    # success = {}
    # thresholds_overlap = np.arange(0, 1.05, 0.05)
    # for tracker in trackers:
    #     print('tracker:' + tracker)
    #     videos = success_ret[tracker]
    #     #print('video num:' + str(len(videos)))
    #     t = []
    #     auc_cal = []
    #     for i in range(8):#8 labels
    #         count = 0
    #         results = {}
    #         for video_name in videos:
    #             if dataset[video_name].attribute[i] == 1:
    #                 results[video_name] = success_ret[tracker][video_name]
    #                 count = count+1
    #         value = [v for k, v in results.items()]
    #         auc_cal.append(np.mean(value))
    #         print('attribute = ' + str(i) + ', num = ' + str(count))
    #     success[tracker] = auc_cal
    #     np.savetxt('./auc.txt', auc_cal, delimiter=' ',newline='\n',fmt = '%.3f')

    #draw_precision_success
    # precision_ret = benchmark.eval_precision()
    # success_ret = benchmark.eval_success()
    # videos = list(dataset.videos.keys())
    # draw_success_precision(success_ret, 'name', videos, 'ALL', precision_ret=precision_ret)

    # video the tracking
    # video = dataset.videos['shot121']
    # video.load_img()
    # video.load_tracker(dataset.tracker_path, 'BACF', True)
    # video.load_tracker(dataset.tracker_path, 'ATOM_format', True)
    # video.show(show_name=True)

if __name__ == '__main__':
    main()
    
