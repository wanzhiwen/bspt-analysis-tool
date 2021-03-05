import os
import numpy as np
import json
# convert the dataset into metadata in json format

def main():
    meta_data = {}
    database_root = 'H:/laboratory/data_checked'
    databases = os.listdir(database_root)
    for database in databases:
        database_path = os.path.join(database_root, database)
        gt_path = os.path.join(database_path, 'anno', 'groundtruth8')
        absent_path = os.path.join(database_path, 'anno', 'absent')
        attribute_path = os.path.join(database_path, 'anno', 'attribute')
        label_path = os.path.join(database_path, 'anno', 'label')
        img_path = os.path.join(database_path, 'sequences')
        videos = os.listdir(img_path)
        videos.sort()
        videos = [title for title in videos if os.path.isdir(os.path.join(img_path, title))]
        for video in videos:
            print('processing ' + video)
            img_names = os.listdir(os.path.join(img_path, video))
            img_names.sort()
            img_names = [title for title in img_names if not title.endswith("txt")]
            gt = np.loadtxt(os.path.join(gt_path, video + '.txt'), delimiter=',')
            gt = gt.tolist()
            absent = np.loadtxt(os.path.join(absent_path, video + '.txt'), delimiter=',', dtype=int)
            absent = absent.tolist()
            attribute = np.loadtxt(os.path.join(attribute_path, video + '.txt'), delimiter=',', dtype=int)
            attribute = attribute.tolist()
            label = np.loadtxt(os.path.join(label_path, video + '.txt'), delimiter=',',dtype=int)
            label = label[:,1:] #the first col in label is meaningless
            label = label.tolist()
            # check
            if not len(img_names)==len(gt)==len(absent)==len(label):
                print('error! in %s,img_names_len:%d,gt_len:%d,label_len:%d,length not match!'%(video,len(img_names), len(gt), len(label)))
                return
            meta_data[video] = {}
            meta_data[video]['video_dir'] = os.path.join(img_path, video)
            meta_data[video]['init_rect'] = gt[0][:]
            meta_data[video]['img_names'] = img_names
            meta_data[video]['gt_rect'] = gt
            meta_data[video]['absent'] = absent
            meta_data[video]['attribute'] = attribute
            meta_data[video]['label'] = label
    with open('test.json','w') as f:
        json.dump(meta_data, f)

if __name__ == '__main__':
    main()