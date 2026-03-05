<h1 align="center">EMVCC</h1>

<p align="center"><strong>Enhanced Multi-View Contrastive Clustering for Hyperspectral Images</strong></p>

<p align="center">
  <b>Fulin Luo</b><sup>1</sup> &nbsp; <b>Yi Liu</b><sup>1</sup> &nbsp; <b>Xiuwen Gong</b><sup>2</sup> &nbsp; <b>Zhixiong Nan</b><sup>1</sup> &nbsp; <b>Tan Guo</b><sup>3</sup><code>∗</code>
</p>


<p align="center">
  <sup>1</sup> College of Computer Science, Chongqing University, China &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup> Faculty of Engineering, The University of Sydney, Australia &nbsp;&nbsp;|&nbsp;&nbsp; <sup>3</sup> Chongqing University of Posts and Telecommunications, China
</p>


<p align="center">
  <small><code>∗</code> Corresponding Author</small>
</p>


<p align="center">
  <a href="https://doi.org/10.1145/3664647.3681600"><img src="https://img.shields.io/badge/ACM%20MM%20'24-10.1145%2F3664647.3681600-0085ca?style=flat-square" alt="DOI"></a>
  &nbsp;
  <a href="https://github.com/YiLiu1999/EMVCC"><img src="https://img.shields.io/badge/Code-EMVCC-green?style=flat-square" alt="Code"></a>
</p>


<p align="center">
  If you find this project helpful, please consider giving it a <strong>⭐ star</strong>!
</p>


---

## 📖 Introduction

Cross-view consensus representation is critical for **hyperspectral image (HSI) clustering**. Existing multi-view contrastive methods suffer from: (1) **False negatives** — contrastive learning may treat similar heterogeneous views as positive pairs and dissimilar homogeneous views as negative pairs; (2) **Clustering-agnostic representation** — self-supervised contrastive features are not designed for clustering. To address this, we propose **EMVCC** (Enhanced Multi-View Contrastive Clustering):

- **Spatial multi-view** — Spectral segmentation builds multi-view data; **spectrum-view** is processed by a **Transformer** to capture global spectral information and enhance the distinction between neighboring samples in spatial views.
- **Self-Supervised Joint Loss (SSJL)** — Constrains consensus representation from multiple perspectives to reduce false negatives:
  - **NT-Xent (contrastive loss)** — Pulls together different views of the same sample and pushes apart different samples.
  - **Feature Enhanced loss (FE)** — Probabilistic contrastive constraint preserves multi-view diversity and brings similar samples closer in semantic space.
  - **Cluster-Friendly loss (CF)** — Aligns view features with high-confidence pseudo-labels so the network learns clustering-friendly features.
- **K-means** updates cluster centers during training; labels are assigned by similarity to cluster centers **without post-processing**.

Experiments on Salinas, Botswana, Indian Pines, and Houston show that EMVCC outperforms state-of-the-art HSI clustering methods.

---

## 🏗️ Method Overview
<img width="2504" height="1136" alt="image" src="https://github.com/user-attachments/assets/6099af7d-e0f8-4267-bd7c-777d657d06a3" />

- **Multi-view construction:** Split spectral channels in half → PCA to 3-D per half → sliding-window patches → data augmentation (crop, flip, rotate, etc.) for spatial views; full spectrum as an extra view.
- **Spectrum-enhanced spatial view:**
  - **Spatial branch:** ResNet extracts spatial features → projection head & classification head.
  - **Spectrum branch:** Transformer extracts global spectral features → projection head & classification head.
  - **Fusion:** Projection features are summed for consensus contrastive representation; classification features are summed for cluster-friendly loss.
- **Loss:** `L = L_NT + α·L_FE + β·L_CF` (α=1, β=0.1 in the paper).

---

## 📂 Repository Structure

```
EMVCC/
├── main.py                 # Training and evaluation entry
├── utils.py                # Dataset, metrics (OA, NMI, AMI, ARI, FMI, Kappa, Purity)
├── losses.py               # NT_Xent, PCLoss (FE), FC_loss (CF)
├── show.py                 # Clustering maps and t-SNE visualization
├── module/
│   ├── EMVCC.py            # Main model (ResNet + Transformer fusion)
│   ├── ResNet.py           # ResNet50 spatial encoder
│   └── Spectral.py        # Transformer spectral encoder
├── dataset/
│   └── create_dataset.py   # Build multi-view H5 from raw HSI .mat
└── cluster/
    └── kmeans.py           # K-means for cluster center update
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YiLiu1999/EMVCC.git
cd EMVCC

# Create environment (Python 3.10 recommended)
conda create -n emvcc python=3.10
conda activate emvcc

# Install dependencies
pip install torch torchvision
pip install h5py scipy numpy scikit-learn tqdm pandas matplotlib
# Optional: pip install thop  # for FLOPs/params
```

---

## 💡 Data Preparation & Usage

### 1. Raw data

Prepare HSI cube and ground-truth labels (`.mat`), e.g.:

- Indian Pines: `Indian_pines_corrected.mat` + `Indian_pines_gt.mat`
- Pavia University: `PaviaU.mat` + `PaviaU_gt.mat`
- Salinas: `Salinas.mat` + `Salinas_gt.mat`
- Botswana: `Botswana.mat` + `Botswana_gt.mat`
- Houston: `HoustonU.mat` + `HoustonU_GT.mat`
- HanChuan: `WHU_Hi_HanChuan.mat` + `WHU_Hi_HanChuan_gt.mat`

### 2. Build H5 multi-view dataset

In `dataset/create_dataset.py`:

- Uncomment the block for your dataset and set `img` / `gt` paths to your `.mat` files.
- Set the output path in `h5py.File(...)` (e.g. `Sa-28-28-230.h5`).
- Run the script. Output H5 must have:
  - `data`: shape compatible with `utils.HsiDataset` (e.g. `(N, 28, 28, b)` — first 6 channels for two spatial views, rest for spectrum).
  - `label`: per-sample class index (0-based).

Adjust `PATCH_SIZE` if you need 28×28 patches (e.g. use 14 for 28×28) or change `reshape` in `utils.py` accordingly.

### 3. Set paths in main.py

- Set `f = h5py.File('...')` and `ture_y = sio.loadmat('...')` to your H5 and ground-truth paths.
- The script sets `c` (number of clusters), `num`, and `b` per dataset; ensure `b` matches the H5 channel count.

### Run training

```bash
# Default: Indian Pines (dataset id 0)
python main.py --dataset 0

# Other datasets: 0=indian, 1=paviau, 2=salinas, 3=botswana, 4=houstonu, 5=hanchuan
python main.py --dataset 2 --epochs 200 --batch_size 128 --anchors 16
```

### Key options

| Option          | Description                                       |
| --------------- | ------------------------------------------------- |
| `--dataset`     | Dataset id (0–5)                                  |
| `--feature_dim` | Projection dimension (default: 128)               |
| `--temperature` | Contrastive loss temperature (default: 0.5)       |
| `--epochs`      | Training epochs (default: 200)                    |
| `--batch_size`  | Batch size (default: 128)                         |
| `--anchors`     | Number of clusters (set automatically by dataset) |
| `--embedding`   | Embedding / cluster center dim (default: 128)     |

**Note:** Default device in `main.py` is `cuda:2`; change it if needed. Best model and logs are saved under `./results/<dataname>/`. Update the save path in `show.py` if you use a different results directory.

---

## 📊 Datasets & Implementation Details

From the paper (Table 1). Same multi-view setup is used for fair comparison.

| Dataset      | Clusters | Samples | Views | Patch size | Bands |
| ------------ | -------- | ------- | ----- | ---------- | ----- |
| Salinas      | 16       | 54,129  | 3     | 28×28×3    | 224   |
| Botswana     | 14       | 3,248   | 3     | 28×28×3    | 145   |
| Indian Pines | 16       | 10,249  | 3     | 28×28×3    | 200   |
| Houston      | 15       | 15,029  | 3     | 28×28×3    | 144   |

**Implementation:** ResNet-50 + Transformer (6 heads); classification head hidden dim 2048, projection head 2048→128; Adam, lr=1e-3, weight_decay=1e-6; α=1, β=0.1.

---

## 🌟 Evaluation

Clustering metrics: **OA**, **Kappa**, **Purity**, **ARI**, **AMI**, **FMI**, **NMI**. Example results (paper Table 2, best in bold):

| Methods   | Salinas OA | Botswana OA | Indian Pines OA | Houston OA |
| --------- | ---------- | ----------- | --------------- | ---------- |
| MFLVC     | 74.20      | 70.67       | 53.17           | 42.08      |
| GCFAgg    | 50.13      | 58.53       | 38.14           | 26.19      |
| ACCMVC    | 73.41      | 70.47       | 55.80           | 45.69      |
| DSCRLE    | 75.25      | 92.73       | 48.88           | 61.29      |
| SDST      | 75.80      | 74.26       | 54.22           | 54.93      |
| **EMVCC** | **77.09**  | **94.40**   | **65.13**       | **70.36**  |




---

## 📜 Citation

If you use this code or the paper in your research, please cite:

```bibtex
@inproceedings{luo2024emvcc,
  title     = {EMVCC: Enhanced Multi-View Contrastive Clustering for Hyperspectral Images},
  author    = {Luo, Fulin and Liu, Yi and Gong, Xiuwen and Nan, Zhixiong and Guo, Tan},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3664647.3681600}
}
```

