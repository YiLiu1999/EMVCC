from torch import nn
import torch.nn.functional as F
import torch


def NT_Xent(out_1, out_2, temperature, batch_size):
    batch_size = out_1.shape[0]
    # 样本级对齐
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def PCLoss(z1, z2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = cos(z1.t(), z2.t().detach())
    L = -loss.mean() + 1  # 调到0-4
    return L


def FC_loss(data, centers, out):
    sim = F.normalize(torch.mm(data, centers.t()), dim=-1)
    label = sim.softmax(dim=1).argmax(dim=1, keepdim=True).reshape(-1)
    CEL = torch.nn.CrossEntropyLoss()
    return CEL(out, label.long())

