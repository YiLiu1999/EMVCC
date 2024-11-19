import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import cv2

# indian
# img = sio.loadmat('Indian_pines_corrected.mat')
# img = img['indian_pines_corrected']
# gt = sio.loadmat('Indian_pines_gt.mat')
# gt = gt['indian_pines_gt']
# img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]

# paviau
# data = sio.loadmat('PaviaU.mat')
# img = data['paviaU']
# gt = data['paviaU_gt']

# hanchuan
# data_mat = sio.loadmat(
#     'WHU_Hi_HanChuan.mat')
# img = data_mat['WHU_Hi_HanChuan']
# gt_mat = sio.loadmat(
#     'WHU_Hi_HanChuan_gt.mat')
# gt = gt_mat['WHU_Hi_HanChuan_gt']

# salinas
# img = sio.loadmat('Salinas.mat')
# img = img['salinas']
# gt = sio.loadmat('Salinas_gt.mat')
# gt = gt['salinas_gt']

# botswana
# img = sio.loadmat('Botswana.mat')
# img = img['Botswana']
# gt = sio.loadmat('Botswana_gt.mat')
# gt = gt['Botswana_gt']

# HoustonU
data = sio.loadmat('HoustonU.mat')
img = data['HoustonU']
gt = data['HoustonU_GT']

# Yancheng
# data = sio.loadmat('data_hsi.mat')
# y1 = sio.loadmat('train_label.mat')
# y2 = sio.loadmat('test_label.mat')
# img = data['data']
# gt = y1['train_label'] + y2['test_label']

# img, gt = img[0:349, 100:200, :], gt[0:349, 100:200]
m, n, b = img.shape
num = np.bincount(gt.reshape(-1))


def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index - PATCH_SIZE, height_index + PATCH_SIZE)
    width_slice = slice(width_index - PATCH_SIZE, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    patch = patch.reshape(-1, patch.shape[0] * patch.shape[1] * patch.shape[2])
    return patch


img1 = img[:, :, :b // 2]
img2 = img[:, :, b // 2:]

img1 = img1.reshape(-1, b // 2)
img2 = img2.reshape(-1, b - b // 2)
img3 = img.reshape(-1, b)

pca = PCA(n_components=3)

reduced_img1 = pca.fit_transform(img1)
img_max, img_min = reduced_img1.max(), reduced_img1.min()
img1 = (reduced_img1 - img_min) / (img_max - img_min)
img1 = img1.reshape(m, n, 3)
# plt.figure()
# plt.imshow(img1)
# plt.show()
reduced_img2 = pca.fit_transform(img2)
img_max, img_min = reduced_img2.max(), reduced_img2.min()
img2 = (reduced_img2 - img_min) / (img_max - img_min)
img2 = img2.reshape(m, n, 3)
# plt.figure()
# plt.imshow(img2)
# plt.show()

reduced_img3 = img3
img_max, img_min = reduced_img3.max(), reduced_img3.min()
img3 = (reduced_img3 - img_min) / (img_max - img_min)
img3 = img3.reshape(m, n, b)

PATCH_SIZE = 16
# 在图像的边缘添加边框
img1 = cv2.copyMakeBorder(img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
img2 = cv2.copyMakeBorder(img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
img3 = cv2.copyMakeBorder(img3, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0, 0, 0))
gt = cv2.copyMakeBorder(gt, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

[mm, nn, bb] = img1.shape

data = []
label = []
gt_index = []

for i in range(PATCH_SIZE, mm - PATCH_SIZE):
    for j in range(PATCH_SIZE, nn - PATCH_SIZE):
        if gt[i, j] == 0:
            continue
        else:
            temp_data1 = Patch(img1, i, j, PATCH_SIZE)
            temp_data2 = Patch(img2, i, j, PATCH_SIZE)
            temp_data3 = Patch(img3, i, j, PATCH_SIZE)
            temp_data1 = temp_data1.reshape(32, 32, 3)
            temp_data2 = temp_data2.reshape(32, 32, 3)
            temp_data3 = temp_data3.reshape(32, 32, b)
            temp_data = np.concatenate((temp_data1, temp_data2), 2)
            temp_data = np.concatenate((temp_data, temp_data3), 2)
            # temp = np.swapaxes(temp_data,1,2)
            # temp_data = np.swapaxes(temp,0,1)
            temp_data = temp_data.reshape(-1)
            data.append(temp_data)
            label.append(gt[i, j] - 1)
            gt_index.append((i - PATCH_SIZE) * n + j - PATCH_SIZE)
# gt_index=np.array(gt_index)
# np.save('ip_index',gt_index)

data = np.array(data)
data = np.squeeze(data)
data = np.uint8(data * 255)
label = np.array(label)
label = np.squeeze(label)

import h5py

# # indian
# f = h5py.File(
#     '/dataset/data/IP-32-32-206.h5',
#     'w')

# # paviau
# f = h5py.File(
#     '/dataset/data/pu-28-28-109.h5',
#     'w')
# yancheng
# f = h5py.File(
#     '/dataset/data/yc-28-28-259.h5',
#     'w')
# Salinas
# f = h5py.File(
#     '/dataset/data/Sa-28-28-230.h5',
#     'w')

# Botswana
# f = h5py.File(
#     '/dataset/data/Bw-28-28-151.h5',
#     'w')

# houstonu
f = h5py.File(
    '/dataset/data/HU-32-32-200.h5',
    'w')

# hanchuan
# f = h5py.File(
#     '/dataset/data/HC-28-28-280.h5',
#     'w')

f['data'] = data
f['label'] = label
f.close()
