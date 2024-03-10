import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}
random_seed = 38383
sample_size = 10000
test_size = 5000

def get_train_file_name(root_dir, flag):
    all_path = []
    id = cate_to_synsetid[flag]
    sub_path = os.path.join(root_dir, id, 'train')
    if not os.path.isdir(sub_path):
        raise Exception("Not a valid path" + sub_path)
    for file in os.listdir(sub_path):
        if not file.endswith('.npy'):
            continue
        all_path.append(file)
    return all_path

class dataloader(Dataset):
    def __init__(self, path, flag):

        self.all_points = []

        all_path= get_train_file_name(path, flag)
        random.Random(random_seed).shuffle(all_path)

        for file in all_path:
            try:
                data = np.load(file) #(15000,3)
            except Exception as e:
                print(e)
                continue
            self.all_points.append(data[np.newaxis, :])
        self.all_points = np.concatenate(self.all_points) #(N, 15000, 3)

        # Normalization
        dim = self.all_points.shape[2]

        self.all_points_mean = self.all_points.reshape(-1, dim).mean(axis = 0).reshape(1,1,dim)
        self.all_points_std = self.all_points.reshape(-1).std(axis = 0).reshape(1,1,1)
        self.all_points = (self.all_points -self.all_points_mean) / self.all_points_std 

        self.train_points = self.all_points[:, :sample_size]
        self.test_points = self.all_points[:, sample_size:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        idxs = np.arange(sample_size)
        target_point = self.train_points[idx]
        train_points = torch.from_numpy(target_point[idxs, :]).float()

        test_points = self.test_points[idx]
        test_idx = np.arange(test_size)
        test_points = torch.from_numpy(test_points[test_idx, :]).float()
        return {
            'idx': idx,
            'train_points':train_points,
            'test_points':test_points
        }


