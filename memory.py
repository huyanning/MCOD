import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import init


class Memory(nn.Module):
    def __init__(self, args, memory_init=None, logit=None):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = args.num_cluster
        self.memory_dim = args.feat_dim
        self.args = args
        if memory_init is not None:
            self.memory_item = memory_init
        else:
            self.memory_item = F.normalize(torch.rand((self.memory_size, self.memory_dim),
                                                        dtype=torch.float, requires_grad=False), dim=1).cuda()
            self.memory_center = torch.rand(self.memory_size, dtype=torch.float).cuda()

        self.att = torch.rand((self.memory_size, self.memory_dim), dtype=torch.float).cuda() 

    def read(self, logit):
        W = F.softmax(logit, dim=1)
        read_item = torch.mm(W, self.memory_item)
        return read_item

    def write(self, queue, W_queue):
        V = torch.softmax(W_queue, dim=1)
        self.memory_item = torch.mm(V.t(), queue)
        self.memory_item = F.normalize(self.memory_item, dim=1)

    def erase_attention(self, W_queue):
        W_queue = torch.softmax(W_queue, dim=1)
        assignment = torch.argmax(W_queue, dim=1).unsqueeze(1).expand(W_queue.shape[0], self.memory_size)
        clusters = torch.arange(self.memory_size).long().cuda()
        mask = assignment.eq(clusters.expand(W_queue.shape[0], self.memory_size))
        att = (1 - mask.sum(dim=0).float()/W_queue.shape[0])
        att =  att.unsqueeze(1).expand(self.memory_size, self.memory_dim)
        self.memory_item = torch.normal(mean=self.memory_item, std=att)



