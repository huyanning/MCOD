"""Dataset utilities."""
import numpy as np
from torchvision.datasets import SVHN, MNIST, CIFAR10, CIFAR100, FashionMNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import ImageFilter
import random
import torch.utils.data as Data
import torch


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        q = self.base_transform1(x)
        k = self.base_transform2(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def load_data_with_outliers(normal, abnormal, p):
    num_abnormal = int(normal.shape[0]*p/(1-p))
    selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)
    data = np.concatenate((normal, abnormal[selected]), axis=0)
    labels = np.zeros((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 1
    return data, labels



def load_cifar10_with_outliers(class_ind, p):
    data_dir = 'cifar10_data/'
    img_train_data = CIFAR10(root=data_dir, train=True, download=True)
    img_test_data = CIFAR10(root=data_dir, train=False, download=True)
    x_train = img_train_data.data.transpose((0, 3, 1, 2))
    y_train = np.array(img_train_data.targets)
    x_test = img_test_data.data.transpose((0, 3, 1, 2))
    y_test = np.array(img_test_data.targets)
    # X = np.concatenate((x_train, x_test), axis=0)
    # Y = np.concatenate((y_train, y_test), axis=0)
    train_data = x_train[y_train.flatten() == class_ind]
    train_label = np.zeros((train_data.shape[0], ), dtype=np.int32)

    if p == 1:
        test_data = x_test
        test_label = np.zeros((test_data.shape[0], ), dtype=np.int32)
        test_label[y_test.flatten() != class_ind] = 1
    else:
        normal_test = x_test[y_test.flatten() == class_ind]
        anomaly_test = x_test[y_test.flatten() != class_ind]
        num_anomaly = int(normal_test.shape[0] * p / (1 - p))
        selected = np.random.choice(anomaly_test.shape[0], num_anomaly, replace=False)
        test_data = np.concatenate((normal_test, anomaly_test[selected]), axis=0)
        test_label = np.zeros((test_data.shape[0], ), dtype=np.int32)
        test_label[len(normal_test):] = 1
    return train_data, train_label, test_data, test_label


def load_mnist_with_outliers(class_ind, p):
    data_dir = 'mnist_data/'
    img_train_data = MNIST(root=data_dir, train=True, download=True)
    img_test_data = MNIST(root=data_dir, train=False, download=True)
    x_train = img_train_data.data
    y_train = img_train_data.targets
    x_test = img_test_data.data
    y_test = img_test_data.targets
    # X = np.concatenate((x_train, x_test), axis=0)
    # Y = np.concatenate((y_train, y_test), axis=0)
    train_data = x_train[y_train.flatten() == class_ind]
    train_label = np.zeros((train_data.shape[0],), dtype=np.int32)

    if p == 1:
        test_data = x_test
        test_label = np.zeros((test_data.shape[0],), dtype=np.int32)
        test_label[y_test.flatten() != class_ind] = 1
    else:
        normal_test = x_test[y_test.flatten() == class_ind]
        anomaly_test = x_test[y_test.flatten() != class_ind]
        num_anomaly = int(normal_test.shape[0] * p / (1 - p))
        selected = np.random.choice(anomaly_test.shape[0], num_anomaly, replace=False)
        test_data = np.concatenate((normal_test, anomaly_test[selected]), axis=0)
        test_label = np.zeros((test_data.shape[0],), dtype=np.int32)
        test_label[len(normal_test):] = 1
    return train_data, train_label, test_data, test_label


def load_fashion_mnist_with_outliers(class_ind, p):
    data_dir = 'fashion_mnist_data/'
    img_train_data = FashionMNIST(root=data_dir, train=True, download=True)
    img_test_data = FashionMNIST(root=data_dir, train=False, download=True)
    x_train = img_train_data.data
    y_train = img_train_data.targets
    x_test = img_test_data.data
    y_test = img_test_data.targets
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    normal = X[Y.flatten() == class_ind]
    abnormal = X[Y.flatten() != class_ind]
    return load_data_with_outliers(normal, abnormal, p)



class Outlier_data(Dataset):
    def __init__(self, data, label, transform=None, target_transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            self.label = self.target_transform(label)
        return data, label

    def __len__(self):
        return len(self.label)


def data_trans(data, running_info):
    channels= 0
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=3)
        channels = 1
    elif len(data.shape) == 4:
        data = data.transpose((0, 2, 3, 1))
        channels = 3
    else:
        print('Input data has a wrong dim! in {}'.format(running_info))
        exit()
    return data, channels


# def get_train_data(data, target, args):
#     train_transform_list = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(32),
#             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
#             #transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             #transforms.RandomResizedCrop(32, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
#             transforms.ToTensor(),
#             #transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010))
#         ])
#     train_transform = TwoCropsTransform(
#         base_transform1=train_transform_list,
#         base_transform2=train_transform_list)
#
#     train_data = Outlier_data(data, target, transform=train_transform)
#     train_loader = Data.DataLoader(dataset=train_data,
#                                    batch_size=args.batch_size,
#                                    shuffle=True,
#                                    pin_memory=True,
#                                    num_workers=args.workers)
#
#     return train_loader

def get_train_data(data, target, args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # set transforms
    augmentation = T.Compose([
        # T.Resize(resize, Image.ANTIALIAS),
        # T.CenterCrop(cropsize),
        T.ToPILImage(),
        T.RandomResizedCrop(32, scale=(0.2, 1.)),
        # T.RandomApply([
        #     T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # normalize
    ])

    eval_augmentation = T.Compose([
        # T.Resize(resize, Image.ANTIALIAS),
        # T.CenterCrop(cropsize),
        T.ToPILImage(),
        T.Resize(32),
        T.ToTensor(),
        # normalize
    ])

    train_transform = TwoCropsTransform(
        base_transform1=eval_augmentation,
        base_transform2=augmentation)

    train_data = Outlier_data(data, target, transform=train_transform)
    if args.gpu is not None:
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=args.workers)

    else:
        sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_loader = Data.DataLoader(dataset=train_data,
                                       sampler=sampler,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=args.workers)

    eval_data = Outlier_data(data, target, transform=eval_augmentation)
    eval_loader = Data.DataLoader(dataset=eval_data,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=args.workers)

    return train_loader, eval_loader


def get_test_data(data, target, args):
    test_transform_list = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_transform = TwoCropsTransform(
        base_transform1=test_transform_list,
        base_transform2=test_transform_list
    )

    test_data = Outlier_data(data, target, transform=test_transform)
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.workers)
    return test_loader
