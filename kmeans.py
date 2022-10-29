import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=0,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        # initial_state = initial_state.cuda()

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().cuda()
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        if torch.isnan(center_shift):
            initial_state = initialize(X, num_clusters)
        # increment iteration
        iteration = iteration + 1
        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster, initial_state, dis


def kmeans_predict(X, cluster_centers, distance='euclidean'):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2):
    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2):
    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)
    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)
    cosine = A_normalized * B_normalized
    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def run_kmeans(x, arg):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'assignment': [], 'centroids': [], 'density': []}

    # intialize faiss clustering parameters
    d = x.shape[1]
    n = x.shape[0]
    k = int(arg.num_cluster)
    x = x.cuda()
    assignment, centroids, dis = kmeans(X=x, num_clusters=k, distance='euclidean')

    # calculate concentration of each cluster
    assignment_a = assignment.unsqueeze(1).expand(n, k)
    density = torch.zeros(k)
    for i in range(k):
        classes = (torch.ones(n, k)*i).long().cuda()
        mask = assignment_a.eq(classes.expand(n, k))
        label_num = mask.sum().float()
        if label_num > 1:
            density[i] = torch.mean((dis * mask) ** 0.5) / torch.log(label_num + 10)
    dmax = torch.max(density)
    density[density == 0] = dmax
    density = density.numpy()
    density = density.clip(np.percentile(density, 10),
                            np.percentile(density, 90))  # clamp extreme values for stability
    density = arg.temp_w * density / density.mean()  # scale the mean to temperature

    # convert to cuda Tensors for broadcast
    centroids = nn.functional.normalize(centroids, p=2, dim=1)
    density = torch.from_numpy(density).cuda()

    results['centroids'] = centroids
    results['density'] = density
    results['assignment'] = assignment
    return results
