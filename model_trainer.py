import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from random import sample
import torch.autograd as autograd
from nt_xent import NTXentLoss

class Model_Trainer(object):
    def __init__(self,
                 E: nn.Module,
                 E_K: nn.Module,
                 D: nn.Module,
                 optimizer_E: torch.optim,
                 optimizer_D: torch.optim,
                 r: int,
                 train_loader,
                 arg):
        self.optimizer_E = optimizer_E
        self.optimizer_D = optimizer_D
        self.E = E
        self.E_K = E_K
        self.D = D
        self.adversarial_loss = torch.nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosim = nn.CosineSimilarity()
        self.ntxent = NTXentLoss(arg.num_cluster, use_cosine_similarity=False, temperature=arg.temp_w)
        self.l1_loss = nn.L1Loss()
        self.arg = arg
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
        queue = {'feature': [], 'embedding': [], 'cluster': []}
        queue['feature'] = torch.zeros((0, self.arg.feat_dim), dtype=torch.float, requires_grad=False).cuda()
        queue['embedding'] = torch.zeros((0, self.arg.low_dim), dtype=torch.float, requires_grad=False).cuda()
        queue['cluster'] = torch.zeros((0, self.arg.num_cluster), dtype=torch.float, requires_grad=False).cuda()

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
        self.momentum_update()
        encode_q, q, w_q = self.E(data[0])
        encode_k, k, w_k = self.E_K(data[1])
        encode_k, k, w_k = encode_k.detach(), k.detach(), w_k.detach()
        N = q.shape[0]
        K = self.queue['embedding'].shape[0]
        
        # embedding NCE
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        l_pos = torch.bmm(q.view(N, 1, -1), k.view(N, -1, 1))
        l_neg = torch.mm(q.view(N, -1), self.queue['embedding'].T.view(-1, K))
        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long).cuda()
        loss1 = self.cross_entropy(logits/self.arg.temp_f, labels)

        # cluster NCE
        w_qn = F.normalize(w_q, dim=0).t()
        w_kn = F.normalize(w_k, dim=0).t()
        loss2 = self.ntxent(w_qn, w_kn)
        loss3 = torch.sum(torch.sum(w_qn, dim=1)**2)/w_q.shape[0]
        self.queue['feature'] = self.queue_data(self.queue['feature'], encode_k)
        self.queue['feature'] = self.dequeue_data(self.queue['feature'], K=self.r)
        self.queue['embedding'] = self.queue_data(self.queue['embedding'], k)
        self.queue['embedding'] = self.dequeue_data(self.queue['embedding'], K=self.r)
        self.queue['cluster'] = self.queue_data(self.queue['cluster'], w_k)
        self.queue['cluster'] = self.dequeue_data(self.queue['cluster'], K=self.r)

        # prototypical contrast
        if epoch >= self.arg.warmup_epoch:
            real_label = torch.ones((data[0].shape[0], 1)).float().cuda()
            fake_label = torch.zeros((data[0].shape[0], 1)).float().cuda()
            read_item = memory.read(w_q)
            #loss4 = self.adversarial_loss(self.D(read_item), real_label)
            loss4 = 0
            loss5 = (torch.sum((encode_q - read_item)**2)).mean()

            #loss4 = torch.mean(torch.bmm(encode_q.view(N, 1, -1).detach(), read_item.view(N, -1, 1)))+self.adversarial_loss(self.D(read_item), real_label)

            #torch.mean((encode_q - read_item)**2)
            loss = loss1 + loss2 + loss3 * 0.05 + loss4 + loss5
            self.optimizer_E.zero_grad()
            loss.backward()
            self.optimizer_E.step()
            # update discriminator
            #self.optimizer_D.zero_grad()
            #real_loss = self.adversarial_loss(self.D(encode_q.detach()), real_label)
            #fake_loss = self.adversarial_loss(self.D(read_item.detach()), fake_label)
            #loss_adv = (real_loss + fake_loss) / 2
            #loss_adv.backward()
            #self.optimizer_D.step()
          
            memory.write(self.queue['feature'], self.queue['cluster'])
            memory.erase_attention(self.queue['cluster'])

        else:
            loss = loss1 + loss2 + loss3 * 0.05
            self.optimizer_E.zero_grad()
            loss.backward()
            self.optimizer_E.step()
            loss4 = 0
            loss5 = 0

        return loss, loss1, loss2, loss3, loss4, loss5
