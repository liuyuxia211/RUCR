import argparse
import os


def data_path() -> str:
    return '/data/FL_data/'


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default=data_path()+'/CIFAR10/')
    parser.add_argument('--uniform_left', type=float, default=0.2)
    parser.add_argument('--uniform_right', type=float, default=0.95)
    parser.add_argument('--feat_loss_arg', type=float, default=0.0)
    parser.add_argument('--crt_ep', type=int, default=0)
    parser.add_argument('--local_bal_ep', type=int, default=0)
    parser.add_argument('--crt_feat_num', type=int, default=0)
    parser.add_argument('--lr_cls_balance', type=float, default=0.01)
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result'))
    parser.add_argument('--crt_batch_size', type=int, default=256)
    parser.add_argument('--times', type=float, default=1.0)

    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)

    args = parser.parse_args()

    return args
