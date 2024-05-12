import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import torch.nn.functional as F



def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data}')
    return dict_per_client


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list, c_id=None):
        self.indices = indices
        self.c_id = c_id

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)


class MixupDataset_norm(Dataset):
    def __init__(self, mean, fs_all: list, args):
        self.data = []
        self.labels = []
        self.means = mean
        self.num_classes = args.num_classes
        self.device = args.device
        self.crt_feat_num = args.crt_feat_num
        self.fs_all = fs_all
        self.fs_len = len(fs_all)
        self.args = args

        self.__mixup_syn_feat_pure_rand_norm__()

    def __mixup_syn_feat_pure_rand_norm__(self):
        num = self.crt_feat_num
        l = self.args.uniform_left
        r_arg = self.args.uniform_right - l
        for cls in range(self.num_classes):
            fs_shuffle_idx = torch.randperm(self.fs_len)
            for i in range(num):
                lam = np.round(l + r_arg * np.random.random(), 2)
                neg_f = self.fs_all[fs_shuffle_idx[i]]
                mixup_f = lam * self.means[cls] + (1 - lam) * F.normalize(neg_f.view(1, -1), dim=1).view(-1)
                self.data.append(mixup_f)
            self.labels += [cls]*num
        self.data = torch.stack(self.data).to(self.device)
        self.labels = torch.tensor(self.labels).long().to(self.device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]
