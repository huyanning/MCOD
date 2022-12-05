# coding=utf-8
import torch.optim as optim
from model import WideResNet
from torchsummary import summary
import torch
import numpy as np
from model_trainer import Model_Trainer
from memory import Memory
import os
from utils import get_class_name_from_index, save_roc_pr_curve_data
from utils import AverageMeter, evaluator
import time
from checkpoint import load_checkpoint, save_checkpoint, copy_state_dict, Logger
from outlier_datasets import get_train_data, get_test_data, data_trans
import sys
from torchvision.utils import save_image


def cake_experiment(load_dataset_fn, dataset_name, p, run_times, c, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)

    running_info = '{}-cake{}-{}-{}-{}'.format(run_times, args.num_cluster, dataset_name, p,
                                               get_class_name_from_index(c, dataset_name))
    model_path = os.path.join(args.resume, running_info)

    train_data, train_label, test_data, test_label = load_dataset_fn(c, p)
    train_data, channels = data_trans(train_data, running_info)
    test_data, _ = data_trans(test_data, running_info)
    train_loader, eval_loader = get_train_data(train_data, train_label, args)
    test_loader = get_test_data(test_data, test_label, args)

    E = WideResNet(depth=10, fea_out=args.low_dim, memory_size=args.num_cluster, widen_factor=4, in_channel=channels)
    E_K = WideResNet(depth=10, fea_out=args.low_dim, memory_size=args.num_cluster, widen_factor=4, in_channel=channels)
    E = torch.nn.DataParallel(E, device_ids=range(torch.cuda.device_count())).cuda()
    E_K = torch.nn.DataParallel(E_K, device_ids=range(torch.cuda.device_count())).cuda()
    summary(E, (channels, 32, 32))
    optimizer = optim.Adam(E.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    memory = Memory(args, memory_init=None)
    load_checkpoint(E_K, optimizer, memory, model_path, args)
    copy_state_dict(E_K.state_dict(), E)

    trainer = Model_Trainer(E, E_K, optimizer, 4096, train_loader, args)
    auroc_buf = []
    prin_buf = []
    prout_buf = []
    # train
    print("train starting.........")
    for epoch in range(args.start_epoch, args.epochs):
        contrastive_loss = AverageMeter()
        regularize_loss = AverageMeter()
        cluster_loss = AverageMeter()
        read_mse_loss = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        for i, (data, _) in enumerate(train_loader):
            data[0] = data[0].cuda()
            data[1] = data[1].cuda()
            save_image(data[0][:100], "%d_dec.png" % epoch, nrow=10, normalize=True)
            loss = trainer.trainer(data, epoch, memory)
            contrastive_loss.update(loss["Contrastive loss"])
            regularize_loss.update(loss["Regularize loss"])
            cluster_loss.update(loss["Cluster loss"])
            read_mse_loss.update(loss["Read_mse loss"])
            batch_time.update(time.time() - end)
            end = time.time()

        print('Current data information:  \t{}'.format(running_info))
        print('Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Contrastive Loss {:.5f}\t'
              'Cluster Loss {:.5f}\t'
              'Regularize Loss {:.5f}\t'
              'Read mse loss {:.5f}\t'
              .format(epoch+1, i + 1, len(train_loader),
                      batch_time.val, batch_time.avg,
                      contrastive_loss.avg, cluster_loss.avg,
                      regularize_loss.avg, read_mse_loss.avg))

        if epoch % 10 == 9 and epoch >= args.warmup_epoch:
            save_checkpoint({
                'current_running_times': run_times,
                'dataset_name': dataset_name,
                'current_class': c,
                'epoch': epoch + 1,
                'memory_item': memory.memory_item,
                'state_dict': E_K.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filepath=model_path)

            error_list = []
            targets_list = []
            for i, (data, targets) in enumerate(test_loader):
                with torch.no_grad():
                    data[0] = data[0].cuda()
                    data[1] = data[1].cuda()
                    encode, embedding, cluster = E(data[0])
                    read_item = memory.read(cluster)
                    error = torch.sum((read_item - encode)**2, dim=1)
                    error_list.append(error.squeeze().cpu().detach().numpy())
                    targets_list.append(targets.detach().cpu())
            error = np.hstack(error_list)
            targets = np.hstack(targets_list)
            roc_auc, pr_auc_norm, pr_auc_anom =\
                evaluator(error, targets, run_times, dataset_name, p, c, "error", args)

            auroc_buf.append(roc_auc)
            prin_buf.append(pr_auc_norm)
            prout_buf.append(pr_auc_anom)
    final_auroc, final_pr_in, final_pr_out = 0, 0, 0
    if len(auroc_buf) > 0:
        auroc_result = np.hstack(auroc_buf)
        prin_result = np.hstack(prin_buf)
        prout_result = np.hstack(prout_buf)
        best_index = np.argmax(auroc_result)
        final_auroc = auroc_result[best_index]
        final_pr_in = prin_result[best_index]
        final_pr_out = prout_result[best_index]
    return [final_auroc, final_pr_in, final_pr_out]








