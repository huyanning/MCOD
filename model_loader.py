import torch
import os
import torch.nn as nn
import copy


class Model_lader(object):
    def __init__(self,
                 E: nn.Module,
                 E_K: nn.Module,
                 D: nn.Module,
                 optimizer_E: torch.optim,
                 optimizer_D: torch.optim,
                 log_train_dir,
                 log_pretrain_dir,
                 arg,
                 memory=None,
                 ):
        self.optimizer_E = optimizer_E
        self.optimizer_D = optimizer_D
        self.E = E
        self.D = D
        self.E_K = E_K
        self.arg = arg
        self.train_dir = log_train_dir
        self.pretrain_dir = log_pretrain_dir
        self.memory = memory
        if not os.path.isdir(arg.dir):
            os.mkdir(arg.dir)

    def load_model(self):
        if os.path.exists(self.train_dir):
            checkpoint = torch.load(self.train_dir)
            self.E.load_state_dict(checkpoint['model_state_dict_E'])
            self.E_K.load_state_dict(checkpoint['model_state_dict_E_K'])
            self.D.load_state_dict(checkpoint['model_state_dict_D'])
            self.optimizer_E.load_state_dict(checkpoint['optimizer_state_dict_E'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
            self.memory = checkpoint['memory']
            init_epoch = checkpoint['epoch'] + 1
            print("Successful loading model! epoch: {0}".format(init_epoch))
        elif os.path.exists(self.pretrain_dir):
            checkpoint = torch.load(self.pretrain_dir)
            self.E.load_state_dict(checkpoint['model_state_dict'])
            self.D.load_state_dict(checkpoint['model_state_dict'])
            self.E_K = copy.deepcopy(self.E)
            init_epoch = 0
            print("Successful loading pretrain model!")
        else:
            init_epoch = 0
            print("no pretrain model exist! Start training directly ")

        return init_epoch

    def update_model(self, E, E_K, D, optimizer_E, optimizer_D, memory, epoch):
        self.E = E
        self.D = D
        self.E_K = E_K
        self.optimizer_E = optimizer_E
        self.optimizer_D = optimizer_D
        self.memory = memory
        checkpoint = {
            'epoch': epoch,
            'model_state_dict_E': self.E.state_dict(),
            'model_state_dict_E_K': self.E_K.state_dict(),
            'model_state_dict_D': self.D.state_dict(),
            'optimizer_state_dict_E': self.optimizer_E.state_dict(),
            'optimizer_state_dict_D': self.optimizer_D.state_dict(),
            'memory': self.memory,
        }
        torch.save(checkpoint, self.train_dir)
