import os
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def save_roc_pr_curve_data(scores, labels, file_path):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def get_class_name_from_index(index, dataset_name):
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
        'mnist':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'svhn':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    }

    return ind_to_name[dataset_name][index]


def save_result(auroc, pr_in, pr_out, result):
    index = np.argmax(auroc)
    final_auroc = auroc(index)
    final_pr_in = pr_in(index)
    final_pr_out= pr_out(index)
    result[0] = final_auroc
    result[1] = final_pr_in
    result[2] = final_pr_out
    return result


def evaluator(predict, target, run_times, dataset_name, p, c, flag, args):
    predict_pos = predict[target == 1]
    predict_neg = predict[target != 1]
    # calculate AUC
    truth = np.concatenate((np.zeros_like(predict_neg), np.ones_like(predict_pos)))
    predict = np.concatenate((predict_neg, predict_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, predict)
    roc_auc = auc(fpr, tpr)

    # PR curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, predict)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # PR curve where "anormal" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -predict, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    print('Current data information:  \t{}-cake{}-{}-{}-{}-{}'.format(run_times, args.num_cluster, dataset_name, p,
                                                                      get_class_name_from_index(c, dataset_name), flag))
    print('AUROC:{}, AUPR-IN:{}, AUPR-OUT:{}'.format(roc_auc, pr_auc_norm, pr_auc_anom))
    return roc_auc, pr_auc_norm, pr_auc_anom


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


