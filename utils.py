import torch,os
import numpy as np
from Model.Resnet8 import *
from Dataset.dataset import *
from Dataset.long_tailed_cifar10 import *


def model_fusion(list_dicts_local_params: list, list_nums_local_data: list):
    # fedavg
    local_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        local_params[name_param] = value_global_param
    return local_params


def cal_norm_mean(args, c_means, c_dis):
    glo_means = dict()
    c_dis = torch.tensor(c_dis).to(args.device)
    total_num_per_cls = c_dis.sum(dim=0)
    for i in range(args.num_classes):
        for c_idx, c_mean in enumerate(c_means):
            if i not in c_mean.keys():
                continue
            temp = glo_means.get(i, 0)
            # normalize the local prototypes, send the direction to the server
            glo_means[i] = temp + F.normalize(c_mean[i].view(1, -1), dim=1).view(-1) * c_dis[c_idx][i]
        if glo_means.get(i) == None:
            continue
        t = glo_means[i]
        glo_means[i] = t / total_num_per_cls[i]
    return glo_means


def get_mean(args, feat_dim, c_infos, nums):
    real_info = dict()
    syn_info = [torch.zeros(feat_dim).to(args.device)]*args.num_classes
    nums = np.array(nums)
    cls_total = np.sum(nums, axis=0)
    for cls in range(args.num_classes):
        for c_idx, c_info in enumerate(c_infos):
            if cls not in c_info.keys():
                continue
            pre = real_info.get(cls, 0)
            real_info[cls] = pre + c_info[cls] * nums[c_idx][cls]
        if real_info.get(cls) == None:
            continue
        temp = real_info.get(cls)
        real_info[cls] = temp / cls_total[cls]

    real_id = []
    for k, v in real_info.items():
        real_id.append(k)
        syn_info[k] = v
    syn_info = torch.stack(syn_info).to(args.device)

    return syn_info


def get_cls_mean_from_feats(feats_all, labels_all):
    feats_all = torch.cat(feats_all, dim=0)
    labels_all = torch.cat(labels_all)
    real_cls = labels_all.unique()
    cls_means = dict()
    for cls in real_cls.tolist():
        cls_feat = feats_all[labels_all == cls]
        cls_means[cls] = cls_feat.mean(dim=0)
    return cls_means


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)