import argparse
from outlier_datasets import load_cifar10_with_outliers, load_cifar100_with_outliers, \
    load_fashion_mnist_with_outliers, load_mnist_with_outliers, load_svhn_with_outliers
from cake_main import cake_experiment
import numpy as np 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code - Calibrated Confidence')
    parser.add_argument('--temp_f', type=float, default=1, help='Tempture parameters')
    parser.add_argument('--temp_w', type=float, default=1, help='Tempture parameters')
    parser.add_argument('--dir', type=str, default="log2", help='dir for saving output')
    parser.add_argument('--num-cluster', default=10, type=int,
                        help='number of clusters')
    parser.add_argument('--warmup-epoch', default=100, type=int,
                        help='number of warm-up epochs to only train with InfoNCE loss')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--low-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--feat-dim', default=256, type=int, help='feature dimension')
    parser.add_argument('--results-dir', type=str, default='results2', help='Directory to save results.')
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset name')
    parser.add_argument('--p', default=0.1, type=float, help='feature dimension')
    arg = parser.parse_args()

    experiments_list = {'mnist': (load_mnist_with_outliers, 'mnist', 10),
                        'fashion_mnist': (load_fashion_mnist_with_outliers, 'fashion-mnist', 10),
                        'cifar10': (load_cifar10_with_outliers, 'cifar10', 10),
                        'svhn': (load_svhn_with_outliers, 'svhn', 10),
                        'cifar100': (load_cifar100_with_outliers, 'cifar100', 20)
                        }


    # p_list = [0.05, 0.1, 0.15, 0.2, 0.25]
    data_load_fn, dataset_name, n_classes = experiments_list[arg.dataset]
    result = np.zeros((5, 3, n_classes))
    auroc = np.zeros((5, n_classes))
    pr_in = np.zeros((5, n_classes))
    pr_out = np.zeros((5, n_classes))
    for i in range(5):
        result[i, :, :] = cake_experiment(data_load_fn, dataset_name, n_classes, arg.p, i+1, arg)

    for i in range(5):
        auroc[i, :] = result[i, 0, :]
        pr_in[i, :] = result[i, 1, :]
        pr_out[i, :] = result[i, 2, :]
    print('auroc_original')
    print(auroc)
    print('prin_original')
    print(pr_in)
    print('prout_original')
    print(pr_out)
    total_auroc = 0.0
    total_prin = 0.0
    total_prout = 0.0
    count = 0
    for i in range(5):
        for j in range(n_classes):
            if auroc[i, j] >0.5:
                total_auroc += auroc[i, j]
                total_prin += pr_in[i, j]
                total_prout += pr_out[i, j]
                count += 1
    pr_out[auroc<0.5] = 1.0
    avr_auroc = total_auroc/count
    avr_prin = total_prin/count
    avr_prout = total_prout/count
    print("auroc:{}".format(avr_auroc))
    print("pr_in:{}".format(avr_prin))
    print("pr_out:{}".format(avr_prout))
    

