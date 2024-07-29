import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE


def Draw_Classification(pred, label, name, acc, scale: float = 4.0, dpi: int = 400):
    colors = ["black", "yellow", "lightgreen", "indigo", "orange", "pink", "peru", "crimson", "aqua", "dodgerblue",
              "slategrey", "b", "red", "darkcyan", "grey", "olive", "gold", "green"]
    indices = np.where(label != 0)
    label[indices] = pred
    # Create a blank RGB image with the same shape as the label
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    # Assign color values to the pixels corresponding to each category
    for i, color in enumerate(colors):
        rgb = np.array(mcolors.to_rgb(color)) * 255
        rgb_label[label == i] = rgb.astype(np.uint8)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)  # Display color images
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)

    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    foo_fig.savefig(
        '/home/results/{}'.format(name) + '/pred_{:.5f}'.format(acc) + '.png', format='png',
        transparent=True, dpi=dpi, pad_inches=0)
    # plt.show()


def Draw_tsne(X, y, acc, dataname, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    X = X.cpu().numpy()
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    labelout = X
    X_tsne_2d = tsne2d.fit_transform(labelout)
    X = X_tsne_2d[:, 0:2]

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    color = ["yellow", "lightgreen", "indigo", "orange", "pink", "peru", "crimson", "aqua", "dodgerblue",
             "slategrey", "b", "red", "darkcyan", "grey", "olive", "green", "gold"]
    for i in range(X.shape[0]):
        if y[i] == 0:
            s0 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[0])
        if y[i] == 1:
            s1 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[1])
        if y[i] == 2:
            s2 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[2])
        if y[i] == 3:
            s3 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[3])
        if y[i] == 4:
            s4 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[4])
        if y[i] == 5:
            s5 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[5])
        if y[i] == 6:
            s6 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[6])
        if y[i] == 7:
            s7 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[7])
        if y[i] == 8:
            s8 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[8])
        if y[i] == 9:
            s9 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[9])
        if y[i] == 10:
            s10 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[10])
        if y[i] == 11:
            s11 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[11])
        if y[i] == 12:
            s12 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[12])
        if y[i] == 13:
            s13 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[13])
        if y[i] == 14:
            s14 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[14])
        if y[i] == 15:
            s15 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[15])
    # plt.xlabel('t-SNE:dimension 1', fontsize=15)
    # plt.ylabel('t-SNE:dimension 2', fontsize=15)
    if title is not None:
        plt.title(title)
    if dataname == 'indian1':
        plt.legend((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15),
                   ("Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
                    "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
                    "Soybean-clean",
                    "Wheat",
                    "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"), loc='best')
    plt.rc('font', family='Times New Roman')
    plt.title('Epoch200', y=-0.1, fontsize=20, fontweight='bold')
    plt.savefig('/home/results/{}'
                .format(dataname) + '/tsne_{:.5f}'.format(acc) + '.pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()
