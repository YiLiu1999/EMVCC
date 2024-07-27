import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import fowlkes_mallows_score as fmi_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class HsiDataset(Dataset):
    def __init__(self, data, label, b, transform):
        self.data = data.reshape(-1, 28, 28, b)
        self.label = label
        self.transform = transform
        self.classes = label.max() + 1

    def __getitem__(self, i):
        img1 = self.data[i, :, :, :3]
        img1 = Image.fromarray(img1)
        img1 = self.transform(img1)
        img2 = self.data[i, :, :, 3:6]
        img2 = Image.fromarray(img2)
        img2 = self.transform(img2)
        img3 = self.data[i, :, :, 6:]
        img3 = img3[15, 15, :]
        img_max, img_min = img3.max(), img3.min()
        img3 = torch.tensor((img3 - img_min) / (img_max - img_min))

        return img1, img2, img3.to(dtype=img1.dtype), self.label[i]

    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28),  #
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),  # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
test_transform = transforms.Compose([
    transforms.ToTensor(),  # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id
        num_classes: total number of classes in your dataset

    Returns: acc and f1-score
    """
    y_true = torch.tensor(y_true) - torch.min(torch.tensor(y_true))
    l1 = list(set(y_true.tolist()))
    num_class1 = len(l1)
    y_pred = torch.tensor(y_pred)
    l2 = list(set(y_pred.tolist()))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred.tolist()))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return

    cost = torch.zeros((num_class1, numclass2), dtype=torch.int32)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # 使用 SciPy 的 linear_sum_assignment 执行 Munkres 算法
    cost_np = cost.numpy()
    row_ind, col_ind = linear_sum_assignment(-cost_np)
    new_predict = torch.zeros(len(y_pred))

    mapping = {}  # 用于建立真实标签到预测标签的映射关系
    for i, c in enumerate(l1):
        c2 = l2[col_ind[i]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
        mapping[c2] = c
    y_true = y_true.cpu()
    acc = metrics.accuracy_score(y_true, new_predict)

    matrix = confusion_matrix(y_true, new_predict)
    # 选择每个簇中最大的数值
    max_cluster_values = np.max(matrix, axis=0)
    # 计算purity
    purity = np.sum(max_cluster_values) / np.sum(matrix)
    ka = kappa(y_true.cpu().numpy(), new_predict.cpu().numpy())
    nmi = nmi_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    ami = ami_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    ari = ari_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    fmi = fmi_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    return acc, new_predict, mapping, purity, ka, nmi, ari, ami, fmi


def eva(y_true, y_pred, c):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, y, mapping, purity, kappa, nmi, ari, ami, fmi = cluster_acc(y_true, y_pred)
    print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ami {:.4f}'.format(ami),
          ', ari {:.4f}'.format(ari),
          ', fmi {:.4f}'.format(fmi), ', kappa {:.4f}'.format(kappa), ', purity {:.4f}'.format(purity))
    return acc, mapping
