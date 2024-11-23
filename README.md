# Federated Learning with Long-Tailed Non-IID Data via Representation Unification and Classifier Rectification

**Abstract:** Prevalent federated learning commonly develops under the assumption that the ideal global class distributions are balanced. In contrast, real-world data typically follows the long-tailed class distribution, where models struggle to classify samples from tail classes. In this paper, we alleviate the issue under the long-tailed data, dissecting the into two aspects: the distorted feature space and the biased classifier. Specifically, we propose the Representation Unification and Classifier Rectification (RUCR), which leverages global unified prototypes to shape the feature space and calibrate the classifier. RUCR aggregates local prototypes (class-wise mean features) extracted by the global model to obtain global unified prototypes. It calibrates the feature space by pulling features within the same class towards corresponding global unified prototypes and pushing the other classes away. Moreover, RUCR utilizes global prototypes to reduce the classifier bias via prototypical mix-up. It generates a balanced virtual feature set by arbitrarily fusing global unified prototypes and local features. The classifier re-training is then conducted on the balanced virtual feature set to rectify the decision boundary and thus alleviate the shifts. Empirical results on CIFAR-10-LT, CIFAR-100-LT, and Tiny-Imagenet-LT datasets validate the superior performance of our proposed method.

### Dependencies

- python 3.8.10 (Anaconda)
- PyTorch 1.10.1
- torchvision 0.11.2
- CUDA 11.4

### Dataset

- CIFAR-10-LT
- CIFAR-100-LT
- Tiny-ImageNet-LT

### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                       | Description                                            |
| ------------------------------ | ------------------------------------------------------ |
| `uniform_left/uniform_right` | Parameters of the Uniform distribution.               |
| `feat_loss_arg`              | Control the magnitude of loss at the feature level. |
| `crt_ep`                     | Number of classfier learning rounds.                 |
| `local_bal_ep`               | Number of local classfier re-training epochs.         |
| `crt_feat_num`               | Number of virtual features per class.                  |
| `lr_cls_balance`             | Learning rate of classifier re-training.               |
| `t`                          | Temperature.                                           |
| `lr_local_training`          | Learning rate of client updating.                      |
| `num_classes`                | Number of classes                                      |
| `num_clients`                | Number of all clients.                                 |
| `num_online_clients`         | Number of participating local clients.                 |
| `num_rounds`                 | Number of communication rounds.                        |
| `num_epochs_local_training`  | Number of local epochs.                                |
| `batch_size_local_training`  | Batch size of local training.                          |
| `non_iid_alpha`              | Control the degree of heterogeneity.                   |
| `imb_factor`                 | Control the degree of imbalance.                       |

### Usage

Here is an example to run RUCR on CIFAR-10-LT with imb_factor=0.01 and non-iid=0.5:

```python
python main.py \
--uniform_left 0.35 \
--uniform_right 0.95 \
--feat_loss_arg 0.15 \
--crt_ep 20 \
--local_bal_ep 50 \
--crt_feat_num 100 \
--lr_cls_balance 0.01 \
--t 0.9 \
--lr_local_training=0.1 \
--num_classes=10 \
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=150 \
--num_epochs_local_training=10 \
--batch_size_local_training=32 \
--non_iid_alpha=0.5 \
--imb_factor=0.01 \
```

### Reference

The code is based on [CReFF](https://github.com/shangxinyi/CReFF-FL).

Here is the raw code: [raw_code](https://github.com/liuyuxia211/raw_code).
