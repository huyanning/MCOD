# coding = coding=utf-8
import torch
import os
import numpy as np
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
# from matplotlib.pyplot import plot, savefig, cla
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os.path as osp


def show(targets, ret1, ret2, epoch):
    target_ids = range(len(set(targets)))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
    
    plt.figure(1, figsize=(12, 10))
    
    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret1[idx, 0], ret1[idx, 1], c=colors[label], label=label)
    
    #for i in range(0, len(targets), 250):
    #    img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
    #    img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5) 
    #    ax.add_artist(AnnotationBbox(img, ret[i]))
    
    plt.legend()
    image_path = osp.join("log", "scatter_embedding_{0}.jpg".format(epoch))
    plt.savefig(image_path, dpi=800)
    plt.close()

    plt.figure(2, figsize=(12, 10))
    
    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret2[idx, 0], ret2[idx, 1], c=colors[label], label=label)
    
    #for i in range(0, len(targets), 250):
    #    img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
    #    img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5) 
    #    ax.add_artist(AnnotationBbox(img, ret[i]))
    
    plt.legend()
    image_path = osp.join("log", "scatter_cluster_{0}.jpg".format(epoch))
    plt.savefig(image_path, dpi=800)
    plt.close()


def detector(test_loader, net, log_dir, epoch, arg):
    print("Starting detecting......")
    E, D = net
    color_map = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
                '#fb9a99', '#FFD700', '#fdbf6f', '#ff7f00',
                '#cab2d6', '#6a3d9a']
    color_map2 = ['Purples', 'Blues', 'Greens', 'Oranges',
                  'YlGn', 'Greys', 'PuRd', 'BuPu',
                  'BuGn', 'RdPu']

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        E.load_state_dict(checkpoint['model_state_dict_E'])
        D.load_state_dict(checkpoint['model_state_dict_D'])
        print("Successful loading Detector!")
    else:
        print("no exit detector in this directory")
        exit()

    predict_tuple = []
    data_tuple = []
    embedding_tuple = []
    label_tuple = []
    feature_tuple = []
    for i, (data, label) in enumerate(test_loader):
        data[0] = data[0].cuda()
        encode, embedding, cluster_label = E(data[0])
        embedding = F.normalize(embedding)
        predict = D(encode)
        predict_tuple.append(predict.squeeze().cpu().detach().numpy())
        data_tuple.append(data[0].squeeze().cpu().detach().numpy())
        embedding_tuple.append(embedding.squeeze().cpu().detach().numpy())
        label_tuple.append(label.squeeze().cpu().detach().numpy())
        feature_tuple.append(encode.squeeze().cpu().detach().numpy())

    data_original = np.vstack(data_tuple)
    predict = np.hstack(predict_tuple)
    embedding = np.vstack(embedding_tuple)
    label = np.hstack(label_tuple)
    feature = np.vstack(feature_tuple)

#    plt.figure(2)
#    embedding = TSNE(n_components=2, random_state=0).fit_transform(embedding)
#    # index = np.random.randint(rows * cols, size=10000)
#    plt.scatter(embedding[:, 0], embedding[:, 1],
#                s=10, c=label, marker='.', alpha=0.9, edgecolors='none')
#    image_path = osp.join(arg.dir, "result_embedding_{}.jpg".format(epoch))
#    plt.savefig(image_path, dpi=800)
#    plt.close()
#
#    plt.figure(3)
#    cluster = TSNE(n_components=2, random_state=0).fit_transform(feature)
#    # index = np.random.randint(rows * cols, size=10000)
#    plt.scatter(cluster[:, 0], cluster[:, 1],
#                s=10, c=label, marker='.', alpha=0.9, edgecolors='none')
#    image_path = osp.join(arg.dir, "feature_cluster_{}.jpg".format(epoch))
#    plt.savefig(image_path, dpi=800)
#    plt.close()

   

    # index = np.random.randint(60000, size=10000)
    # ret1 = TSNE(n_components=2, random_state=0).fit_transform(embedding[index])
    # ret2 = TSNE(n_components=2, random_state=0).fit_transform(cluster_label[index])
    # show(label[index], ret1, ret2, epoch)

#    plt.figure(1)
#    plt.cla()
#    for j in range(10):
#        tmp = np.argwhere(label[index] == j)
#        plt.scatter(show[tmp, 0], show[tmp, 1], c=color_map[j], label='Class:{0}'.format(j), marker='.', edgecolors='none',  alpha=0.2)
#
#    plt.legend(fontsize='xx-small', loc=4)
#    image_path = osp.join("log", "scatter_encode_{0}.jpg".format(epoch))
#    plt.savefig(image_path, dpi=800)
#    plt.close()
   

