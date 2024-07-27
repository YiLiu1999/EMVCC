import torch


class KMEANS:
    def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cpu")):
        # self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # Randomly choose the initial center point, want to faster convergence can be borrowed from sklearn in the kmeans++ initialization method
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        # print(init_row.shape)    # shape 10
        init_points = x[init_row]
        # print(init_points.shape) # shape (10, 2048)
        self.centers = init_points
        while True:
            # Cluster Marker
            self.nearest_center(x)
            # Updating the center point
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        return self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        # print(labels.shape)  # shape (250000)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        # print(dists.shape)   # shape (0, 10)
        for i, sample in enumerate(x):
            # print(self.centers.shape) # shape(10, 2048)
            # print(sample.shape)       # shape 2048
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            # print(dist.shape)         # shape 10
            labels[i] = torch.argmin(dist)
            # print(labels.shape)       # shape 250000
            # print(labels[:10])
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
            # print(dists.shape)        # shape (1,10)
            # print('*')
        self.labels = labels  # shape 250000
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists  # 250000, 10
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)  # shape (0, 250000)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))  # 10, 2048
        self.centers = centers  # shape (10, 2048)

    def representative_sample(self):
        # It is more intuitive to find the sample closest to the center point as a representative sample for clustering
        # print(self.dists.shape)
        self.representative_samples = torch.argmin(self.dists, 1)
        # print(self.representative_samples.shape)  # shape 250000
        # print('*')
        return self.representative_samples
