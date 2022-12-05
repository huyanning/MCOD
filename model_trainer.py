import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import NTXentLoss, InfoNCE

class Model_Trainer(object):
    def __init__(self,
                 E: nn.Module,
                 E_K: nn.Module,
                 optimizer: torch.optim,
                 r: int,
                 train_loader,
                 args):
        self.optimizer = optimizer
        self.E = E
        self.E_K = E_K
        self.NCE_loss = InfoNCE(args)
        self.cosim = nn.CosineSimilarity()
        self.ntxent = NTXentLoss(args.num_cluster, use_cosine_similarity=False, temperature=args.temp_w)
        self.l1_loss = nn.L1Loss()
        self.args = args
        self.r = r
        self.queue = self.initialize_queue(train_loader)

    def momentum_update(self, beta=0.999):
        param_k = self.E_K.state_dict()
        param_q = self.E.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta * param_k[n].data + (1 - beta) * q.data)
        self.E_K.load_state_dict(param_k)

    def queue_data(self, data, k):
        return torch.cat([data, k], dim=0)

    def dequeue_data(self, data, K):
        if len(data) > K:
            return data[-K:]
        else:
            return data

    def initialize_queue(self, train_loader):
        queue = {'feature': torch.zeros((0, self.args.feat_dim), dtype=torch.float, requires_grad=False).cuda(),
                 'embedding': torch.zeros((0, self.args.low_dim), dtype=torch.float, requires_grad=False).cuda(),
                 'cluster': torch.zeros((0, self.args.num_cluster), dtype=torch.float, requires_grad=False).cuda()}

        for i, (data, target) in enumerate(train_loader):
            data[1] = data[1].cuda()
            enco_k, k, w_k = self.E_K(data[1])
            enco_k, k, w_k = enco_k.detach(), k.detach(), w_k.detach()
            k = F.normalize(k, dim=1)
            queue['feature'] = self.queue_data(queue['feature'], enco_k)
            queue['feature'] = self.dequeue_data(queue['feature'], K=10)
            queue['embedding'] = self.queue_data(queue['embedding'], k)
            queue['embedding'] = self.dequeue_data(queue['embedding'], K=10)
            queue['cluster'] = self.queue_data(queue['cluster'], w_k)
            queue['cluster'] = self.dequeue_data(queue['cluster'], K=10)
            break
        return queue

    def trainer(self, data, epoch, memory):
        self.E.train()
        self.E_K.train()
        self.momentum_update()
        encode_q, q, w_q = self.E(data[0])
        encode_k, k, w_k = self.E_K(data[1])
        encode_k, k, w_k = encode_k.detach(), k.detach(), w_k.detach()
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # embedding NCE
        contrastive_loss = self.NCE_loss(q, k, self.queue['embedding'])
        # cluster NCE
        w_qn = F.normalize(w_q, dim=0).t()
        w_kn = F.normalize(w_k, dim=0).t()
        cluster_loss = self.ntxent(w_qn, w_kn)
        regularize_loss = torch.sum(torch.sum(w_qn, dim=1) ** 2) / w_q.shape[0] * 0.05

        self.queue['feature'] = self.queue_data(self.queue['feature'], encode_k)
        self.queue['feature'] = self.dequeue_data(self.queue['feature'], K=self.r)
        self.queue['embedding'] = self.queue_data(self.queue['embedding'], k)
        self.queue['embedding'] = self.dequeue_data(self.queue['embedding'], K=self.r)
        self.queue['cluster'] = self.queue_data(self.queue['cluster'], w_k)
        self.queue['cluster'] = self.dequeue_data(self.queue['cluster'], K=self.r)

        # prototypical contrast
        if epoch >= self.args.warmup_epoch:
            read_item = memory.read(w_q)
            read_mse = (torch.sum((encode_q - read_item)**2)).mean()
            #loss5 = torch.mean((encode_q - read_item)**2)
            loss = contrastive_loss + cluster_loss + regularize_loss + read_mse
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
          
            memory.write(self.queue['feature'], self.queue['cluster'])
            memory.erase_attention(self.queue['cluster'])

        else:
            loss = contrastive_loss + cluster_loss + regularize_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            read_mse = 0

        loss_dic = {"Contrastive loss": contrastive_loss,
                    "Cluster loss": cluster_loss,
                    "Regularize loss": regularize_loss,
                    "Read_mse loss": read_mse}
        return loss_dic
