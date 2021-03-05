import argparse
from tools.database.soccer import SoccerDataset
from tools.evaluation.opebenchmark import OPEBenchmark
from glob import glob
import os
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
    dataset.videos['shot121'].load_tracker(dataset.tracker_path, 'BACF', True)
    #print(dataset.videos['shot121'].pred_trajs['BACF'])
    benchmark = OPEBenchmark(dataset)
    success_ret = benchmark.eval_precision()
    print(success_ret['BACF']['shot121'])

if __name__ == '__main__':
    main()
    
