import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .ResNet import ResNet50, ResNet34
from .Spectral import transformer, LSTM


class EMVCC(nn.Module):
    def __init__(self, feature_dim, b, device, args):
        super(EMVCC, self).__init__()
        self.n_anchors = args.anchors
        # self.encoder1 = ResNet34(device, num_classes)
        # self.encoder2 = LSTM(3, num_classes, b)

        self.encoder3 = ResNet50(device, feature_dim)
        self.encoder4 = transformer(b, 4 * feature_dim, feature_dim, 0, 0, 6)

        self.weight = Parameter(torch.FloatTensor(128, 128), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight)
        self.act = nn.Tanh()
        self.device = device
        self.s = nn.Sigmoid()
        self.anchor_centers = Parameter(torch.Tensor(args.anchors, args.embedding), requires_grad=True)

    def update_anchors(self, data):
        sim = F.normalize(torch.mm(data, self.anchor_centers.t()), dim=-1)
        label = sim.softmax(dim=1).argmax(dim=1, keepdim=True).reshape(-1)
        for i in range(self.n_anchors):
            indices = label == i
            if indices.sum() > 0:
                self.anchor_centers[i].data = torch.mean(data[indices], dim=0)

    def forward(self, input, pot):
        # a
        # feature1, out1 = self.encoder1(input)
        # feature2, out2 = self.encoder2(pot)

        # b
        feature3, out3 = self.encoder3(input)
        feature4, out4 = self.encoder4(pot)

        # fusion
        # feature_a, out_a = (feature1 + feature2) / 2, (out1 + out2) / 2
        feature_b, out_b = (feature3 + feature4) * 1e2 / 2, (out3 + out4) / 2

        return feature_b, out_b
