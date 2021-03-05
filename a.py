import numpy as np
import os

def main():
    root = 'D:\LTMU_data\ATOM_MU'
    label_path = os.path.join(root, 'label')
    result_path = os.path.join(root, 'results_update')
    videos = os.listdir(result_path)
    videos.sort()
    videos = [title for title in videos if os.path.isfile(os.path.join(result_path, title))]
    var = np.zeros((8,4))
    for video in videos:
        label = np.loadtxt(os.path.join(label_path, video), delimiter=',')
        label = label[:,1:]
        result = np.loadtxt(os.path.join(result_path, video), delimiter=',')
        if len(label) != len(result):
            print('length not match')
            return
        for i in range(len(label)):
            for j in range(8):
                if label[i, j] == 1 and result[i] == 1:
                    var[j, 0] = var[j, 0] + 1
                elif label[i, j] == 1 and result[i] == 0:
                    var[j, 1] = var[j, 1] + 1
                elif label[i, j] == 0 and result[i] == 1:
                    var[j, 2] = var[j, 2] + 1
                elif label[i, j] == 0 and result[i] == 0:
                    var[j, 3] = var[j, 3] + 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                else:
                    print('data error')
                    return
                    
    print(var)


if __name__ == '__main__':
    main()