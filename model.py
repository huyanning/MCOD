import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, fea_out, memory_size, widen_factor=1, dropRate=0.0, in_channel=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.fea_out = fea_out
        self.memory_size = memory_size
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channel, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        # self.final_act = nn.Softmax(dim=1)
        self.contrast_layer = nn.Linear(nChannels[3], self.fea_out)
        #self.contrast_layer = nn.Sequential(
        #    nn.Linear(nChannels[3],128),
        #    nn.ReLU(),
        #    nn.Linear(128, self.fea_out)
        #)
        self.cluster_layer = nn.Sequential(
            # nn.Linear(self.fea_out, self.fea_out),
            # nn.ReLU(),
            nn.Linear(self.fea_out, self.memory_size),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = 1. / np.sqrt(m.weight.data.size(1))
                m.weight.data.uniform_(-n, n)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        features = out.view(-1, self.nChannels)
        embeddings = self.contrast_layer(self.relu(features))
        clusters = self.cluster_layer(self.relu(embeddings))

        return features, embeddings, clusters


class Discriminator(nn.Module):
    def __init__(self, fea_in, memory_size):
        super(Discriminator, self).__init__()
        self.fea_in = fea_in
        self.memory_size = memory_size
        self.model = nn.Sequential(
            nn.Linear(self.fea_in, self.memory_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.memory_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, fea_out, memory_size):
        super(Encoder, self).__init__()
        self.fea_out = fea_out
        self.memory_size = memory_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),   #14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),  # 3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),  # 1
            nn.LeakyReLU(inplace=True),
            Flatten(),
        )
        self.constrative_layer = nn.Sequential(
            nn.Linear(256, self.fea_out),
        )
        self.matte_layer = nn.Sequential(
            nn.Linear(self.fea_out, self.memory_size),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        encode = self.encoder(x)
        feature = self.constrative_layer(encode)
        weight = self.matte_layer(feature)
        return F.normalize(encode), feature, weight
