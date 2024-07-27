import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import scipy.io as sio
from utils import eva
from tqdm import tqdm
from losses import NT_Xent, PCLoss, FC_loss
from module.EMVCC import EMVCC as Model
import numpy as np
import random
import utils
import show
from datetime import datetime
from cluster.kmeans import KMEANS
import warnings
import torch.nn.functional as F
from thop import profile
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
device = torch.device("cuda:{}".format(2) if torch.cuda.is_available() else "cpu")


def pre_train(net, memory_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data1, data2, pos_3, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data1, data2, pos_3 = data1.to(device), data2.to(device), pos_3.to(device)
            feature1, out_1a = net(data1, pos_3)
            feature2, out_2a = net(data2, pos_3)
            feature = feature1 + feature2
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    return feature_bank


def train(net, data_loader, train_optimizer, anchor_id, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, pos_3, target in train_bar:
        pos_1, pos_2, pos_3 = pos_1.to(device), pos_2.to(device), pos_3.to(device)
        # print(pos0_0.shape)
        feature1, out_1a = net(pos_1, pos_3)
        feature2, out_2a = net(pos_2, pos_3)
        feature = feature1 + feature2
        out = out_2a + out_1a
        # net.update_anchors(feature)
        loss_NT = NT_Xent(out_1a, out_2a, temperature, batch_size)
        # loss_NT = Loss_X(out_1a, out_2a)
        # loss_NT = Loss_L(out_1a, out_2a)
        # loss_NT = Loss_M(out_1a, out_2a)
        loss_PC = PCLoss(out_1a, out_2a)
        # loss_kl = kl(out_1a, model) + kl(out_2a, model)
        # loss_dist = _loss(out_1a, anchor_id, net.anchor_centers, device) + _loss(out_2a, anchor_id,
        #                                                                          net.anchor_centers, device)
        loss_FC = FC_loss(feature, net.anchor_centers, out)

        loss = loss_NT + loss_PC + loss_FC * 0.1  # 总体
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num


def test(epoch, net, memory_data_loader, best_score):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data1, data2, pos_3, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data1, data2, pos_3 = data1.to(device), data2.to(device), pos_3.to(device)
            feature1, out_1a = net(data1, pos_3)
            feature2, out_2a = net(data2, pos_3)
            feature = feature1 + feature2
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        Target = torch.tensor(memory_data_loader.dataset.label, device=feature_bank.device)
        # 1 ada_spectral_cluster
        # label = spectral_clustering_adaptive(feature_bank, c)

        # 2 anchor_cluster
        sim_matrix = F.normalize(torch.mm(feature_bank, model.anchor_centers.t()), dim=-1)
        label = sim_matrix.softmax(dim=1).argmax(dim=1, keepdim=True).reshape(-1)
        acc, map = eva(Target, label, c)

        model0 = KMEANS(args.anchors, max_iter=20, verbose=False, device=device)
        anchor_id = model0.fit(feature_bank)
        model.anchor_centers.data = model0.centers
        transformed_arr = np.array([map[element] for element in label.cpu().numpy()])
        label = transformed_arr
    return acc, label, feature_bank


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    dataset = {
        0: "indian",
        1: "paviau",
        2: "salinas",
        3: "botswana",
        4: "houstonu",
        5: "hanchuan"
    }
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Train EMVCC')
    parser.add_argument('--dataset', type=int, default='0', help='dataset id')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--anchors', default=16, type=int, help='Number of anchor')
    parser.add_argument('--embedding', default=128, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, help='Learning rate update method')
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataname = dataset[args.dataset]
    setup_seed(3307)
    if dataname == 'indian':
        c, num, b = 16, 10249, 206
        f = h5py.File(
            '/home/data/IP-28-28-206.h5', 'r')
        ture_y = sio.loadmat('/home/Indian_pines_gt.mat')['indian_pines_gt']
    elif dataname == 'paviau':
        c, num, b = 9, 42776, 109
        f = h5py.File(
            '/home/data/pu-28-28-109.h5', 'r')
        ture_y = sio.loadmat('/home/PaviaU_gt.mat')[
            'paviaU_gt']
    elif dataname == 'salinas':
        # 29 97.44%
        c, num, b = 16, 54129, 230
        f = h5py.File('/home/data/Sa-28-28-230.h5', 'r')
        ture_y = sio.loadmat('/home/Salinas_gt.mat')[
            'salinas_gt']
    elif dataname == 'botswana':
        # Bw 14 #26
        c, num, b = 14, 3248, 151
        f = h5py.File('/home/data/Bw-28-28-151.h5', 'r')
        ture_y = sio.loadmat('/home/Dataset/Botswana_gt.mat')[
            'Botswana_gt']
    elif dataname == 'houstonu':
        # HU #27 #
        c, num, b = 15, 15029, 150
        f = h5py.File('/home/data/HU-28-28-200.h5', 'r')
        ture_y = sio.loadmat('/home/Dataset/HoustonU.mat')[
            'HoustonU_GT']
    else:
        # HU #27 #
        c, num, b = 16, 257530, 280
        f = h5py.File('/home/data/HC-28-28-280.h5', 'r')
        ture_y = sio.loadmat('/home/WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
    args.anchors = c
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    train_data = utils.HsiDataset(data, label, b, transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=False)
    test_data = utils.HsiDataset(data, label, b, transform=utils.train_transform)
    test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    memory_data = utils.HsiDataset(data, label, b, transform=utils.test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Model(args.feature_dim, b - 6, device, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # training loop
    results = {'train_loss': [], 'test_acc': []}
    save_name_pre = '{}'.format(current_time)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_score = 0.0

    # Pre_training and Initialize the clustering center
    feature_bank = pre_train(model, train_loader)
    model0 = KMEANS(args.anchors, max_iter=20, verbose=False, device=device)
    anchor_id = model0.fit(feature_bank)
    model.anchor_centers.data = model0.centers
    print('finish_init_anchor_centers')

    gt = train_data.label
    best_label = torch.zeros(num)
    best_feature = torch.zeros(num, 256)
    best_acc = 0

    milestones = [50, 100, 150]
    if args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    input = torch.randn(1, 3, 32, 32).to(device)
    b = torch.randn(1, b - 6).to(device)
    flops, params = profile(model, inputs=(input, b))
    print('flops:', flops / (1024 * 1024 * 1024))
    print('params:', params / 1000000)

    # Start training
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, anchor_id, epoch)
        results['train_loss'].append(train_loss)
        acc, pred, feature = test(epoch, model, memory_loader, best_score)
        # scheduler.step()
        results['test_acc'].append(acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('./results/{}/{}_statistics.csv'.format(dataname, save_name_pre), index_label='epoch')
        if acc > best_acc:
            best_epoch = epoch
            best_label = pred
            best_acc = acc
            best_feature = feature
            torch.save(model.state_dict(), './results/{}/{}_model.pth'.format(dataname, save_name_pre))

    if best_label is not None:
        # Make sure the save path exists
        save_path = './results/{}'.format(dataname)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Convert best_label to string format
        best_label_str = '\n'.join(
            map(str, best_label.tolist() if isinstance(best_label, torch.Tensor) else best_label))

        # Write to text file
        with open(os.path.join(save_path, 'best_label.txt'), 'w') as file:
            file.write(best_label_str)
    print('====>best accuracy:', best_acc)

    # Visualization of clustering results
    show.Draw_Classification(best_label + 1, ture_y, dataname, best_acc)

    # T-SNE Visualization
    show.Draw_tsne(best_feature, gt, best_acc, dataname, title=None)
