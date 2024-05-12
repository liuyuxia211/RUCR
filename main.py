import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch import eq, no_grad
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import random
from Model.Resnet8 import ResNet_cifar_rucr
from Dataset.dataset import *
from Dataset.sample_dirichlet import clients_indices
from Dataset.long_tailed_cifar10 import train_long_tail
from tqdm import tqdm
from utils import *
from options import args_parser


class Global(object):
    def __init__(self,
                 device: str,
                 args,
                 ):
        self.device = device
        self.num_classes = args.num_classes
        self.criterion = CrossEntropyLoss().to(args.device)
        self.syn_model = ResNet_cifar_rucr(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None,
                                           freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        self.feature_net = nn.Linear(256, args.num_classes).to(args.device)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(
                    dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / \
                sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return copy.deepcopy(self.syn_model.state_dict())

    def cal_norm_mean(self, c_means, c_dis):
        glo_means = dict()
        c_dis = torch.tensor(c_dis).to(self.device)
        total_num_per_cls = c_dis.sum(dim=0)
        for i in range(self.num_classes):
            for c_idx, c_mean in enumerate(c_means):
                if i not in c_mean.keys():
                    continue
                temp = glo_means.get(i, 0)
                # normalize the local prototypes, send the direction to the server
                glo_means[i] = temp + \
                    F.normalize(c_mean[i].view(1, -1),
                                dim=1).view(-1) * c_dis[c_idx][i]
            if glo_means.get(i) == None:
                continue
            t = glo_means[i]
            glo_means[i] = t / total_num_per_cls[i]
        return glo_means


class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int):
        args = args_parser()
        self.args = args
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(args.device)

        self.local_model = ResNet_cifar_rucr(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None,
                                             freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(),
                             lr=args.lr_local_training)
        self.pre_model = ResNet_cifar_rucr(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None,
                                           freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)

        self.cos_sim = nn.CosineSimilarity(dim=1).to(args.device)
        self.log_max = nn.LogSoftmax(dim=1).to(args.device)
        self.nll = nn.NLLLoss().to(args.device)
        self.soft_max = nn.Softmax(dim=1)

        self.feat_dim = self.local_model.classifier.in_features
        self.l_deno = args.num_rounds * args. num_epochs_local_training

        self.g_epoch = 0

    def local_train(self, args, global_params):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        log_ce, log_feat = 0, 0
        for i in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                f, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)
                log_ce += loss
                feat_loss = self.bal_simclr_imp(f, labels)
                log_feat += feat_loss
                loss += (args.feat_loss_arg * feat_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print("Com: %d/%d celoss = %0.3f featloss = %0.3f" %
              (self.g_epoch, args.num_rounds, log_ce, log_feat))
        return self.local_model.state_dict()

    def bal_simclr_imp(self, f, labels):
        f_norm = F.normalize(f, dim=1)
        # cos sim
        sim_logit = f_norm.mm(self.cls_syn_c_norm.T)
        sim_logit_real = sim_logit.index_fill(1, self.fake_id_tenosr, 0)
        # temperature
        sim_logit_tau = sim_logit_real.div(self.args.t)
        # cls ratio
        src_ratio = self.cls_ratio[labels].log() * self.args.times
        add_src = torch.scatter(torch.zeros_like(sim_logit), 1, labels.unsqueeze(1), src_ratio.view(-1, 1))
        f_out = sim_logit_tau + add_src
        loss = self.criterion(f_out, labels)
        return loss

    def get_local_centro(self, args):
        g_m = self.pre_model
        data_loader = DataLoader(dataset=self.data_client,
                                 batch_size=args.batch_size_local_training,
                                 shuffle=True)

        global_feats_all, labels_all = [], []
        with torch.set_grad_enabled(False):
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                global_feat, _ = g_m(images)
                global_feats_all.append(global_feat.data.clone())
                labels_all.append(labels)
        cls_means = get_cls_mean_from_feats(global_feats_all, labels_all)
        syn_c = [torch.zeros(self.feat_dim).to(self.device)]*args.num_classes
        real_id, fake_id = [], []
        for k, v in cls_means.items():
            real_id.append(k)
            syn_c[k] = v
        for i in range(args.num_classes):
            if i in real_id:
                continue
            fake_id.append(i)
        syn_c = torch.stack(syn_c).to(self.device)
        self.fake_id_tenosr = torch.tensor(
            fake_id, dtype=torch.int64).to(self.device)
        feats_all = torch.cat(global_feats_all, dim=0)
        labels_all = torch.cat(labels_all)
        return cls_means, feats_all

    def local_crt(self, glo_means, fs_all, args):
        for param_name, param in self.local_model.named_parameters():
            if 'classifier' not in param_name:
                param.requires_grad = False

        crt_dataset = MixupDataset_norm(glo_means, fs_all, args)
        self.local_model.eval()
        temp_optimizer = SGD(self.local_model.classifier.parameters(), lr=args.lr_cls_balance)
        for i in range(args.local_bal_ep):
            crt_loader = DataLoader(dataset=crt_dataset,
                                    batch_size=args.crt_batch_size,
                                    shuffle=True)
            for feat, cls in crt_loader:
                feat, cls = feat.to(self.device), cls.to(self.device)
                outputs = self.local_model.classifier(feat)
                loss = self.criterion(outputs, cls)
                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()
        return copy.deepcopy(self.local_model.classifier.state_dict())

    def get_cls_ratio(self, global_num=None):
        temp = global_num.clone().float().detach().to(args.device)
        self.cls_ratio = temp / temp.sum()


def CReFF():
    args = args_parser()
    path = args.save_path
    create_if_not_exists(path)
    print(path)
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
        ))
    random_state = np.random.RandomState(args.seed)
    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if args.num_classes == 10:
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)

    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, _, list_label2indices_train_new = train_long_tail(copy.deepcopy(
        list_label2indices), args.num_classes, args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(
        list_label2indices_train_new), args.num_classes, args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(
        data_local_training, list_client2indices, args.num_classes)
    global_model = Global(device=args.device, args=args,)

    feat_dim = global_model.syn_model.classifier.in_features
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    re_trained_acc = []
    best_acc, best_model_params = 0, None
    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        clients, c_means, c_dis, ccvr_means = [], [], [], []
        real_clients = copy.deepcopy(online_clients)

        if args.crt_ep != 0 and args.crt_ep >= args.num_rounds - r:
            real_clients = total_clients
        c_means, c_fs = [], []

        print(str(online_clients))
        # global unified prototypes
        for client in real_clients:
            indices2data = Indices2Dataset(data_local_training)
            indices2data.load(list_client2indices[client], client)
            data_client = indices2data
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            local_model.g_epoch = r

            if client in online_clients:
                list_nums_local_data.append(len(data_client))
                clients.append(local_model)
                c_dis.append(original_dict_per_client[client])

            local_model.pre_model.load_state_dict(copy.deepcopy(global_params))
            real_mean, c_f = local_model.get_local_centro(args)
            ccvr_means.append(real_mean)
            if client in online_clients:
                c_means.append(real_mean)
                c_fs.append(c_f)
        # global unified prototypes
        syn_mean = get_mean(args, feat_dim, c_means, c_dis)
        global_dis = torch.tensor(c_dis).sum(0).to(args.device)

        # local training - representation leanring
        for c_id, l_model in enumerate(clients):
            l_model.get_cls_ratio(global_dis)
            l_model.cls_syn_c = syn_mean
            l_model.cls_syn_c_norm = F.normalize(syn_mean, dim=1)
            local_params = l_model.local_train(args, copy.deepcopy(global_params))
            list_dicts_local_params.append(copy.deepcopy(local_params))


        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        eval_params = copy.deepcopy(fedavg_params)


        # local training - classifer leanring
        if args.crt_ep >= args.num_rounds - r:
            norm_means = global_model.cal_norm_mean(copy.deepcopy(ccvr_means), original_dict_per_client)
            mixup_cls_params = []
            for c_id, l_model in enumerate(clients):
                mixup_cls_param = l_model.local_crt(norm_means, c_fs[c_id], args)
                mixup_cls_params.append(mixup_cls_param)
            mixup_classifier = model_fusion(mixup_cls_params, list_nums_local_data)
            for name_param in reversed(eval_params):
                if name_param == 'classifier.bias':
                    eval_params[name_param] = mixup_classifier['bias']
                if name_param == 'classifier.weight':
                    eval_params[name_param] = mixup_classifier['weight']
                    break

        # global eval
        one_re_train_acc = global_model.global_eval(eval_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
    
        if one_re_train_acc >= best_acc:
            best_acc = one_re_train_acc
            best_model_params = copy.deepcopy(eval_params)
        if r % 10 == 0:
            print(re_trained_acc)
            print(max(re_trained_acc))

    torch.save(best_model_params, path+'/best_model_param.pth')
    print(re_trained_acc)
    print(max(re_trained_acc))
    print(path)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    CReFF()
