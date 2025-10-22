
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_samples
import umap

# 读取原始数据
X = np.load(r"PNW_dataset/pnw_x_test.npy")  # shape (1720, 3, 15000)
y_test = np.load(r"PNW_dataset/pnw_y_test.npy")  # shape (1720,)

# 1️⃣ 截取 4800:10500
X_cut = X[:, :, 4800:10500]  # shape -> (1720, 3, 5700)

# 2️⃣ 通道最后一维
X_cut = np.transpose(X_cut, (0, 2, 1))  # shape -> (1720, 5700, 3)

# 3️⃣ 插值到长度 3750
N, L_old, C = X_cut.shape
L_new = 3750
X_resampled = np.empty((N, L_new, C), dtype=X_cut.dtype)

old_indices = np.arange(L_old)
new_indices = np.linspace(0, L_old-1, L_new)

for i in range(N):
    for ch in range(C):
        f = interp1d(old_indices, X_cut[i,:,ch], kind='linear')
        X_resampled[i,:,ch] = f(new_indices)

print("✅ Resampled data shape:", X_resampled.shape)  # (1720, 3750, 3)

x_test_n = utils.norml(X_resampled)
x_test_stft = utils.batch_stft(x_test_n, nfft=128, overlap=0.7)

# Resize input to 64x64
def resize_stft(x_in):
    batch_num = x_in.shape[0]
    resized_input = np.zeros((batch_num, 64, 64, 3), dtype=np.float32)
    for i in range(batch_num):
        resized_image = cv2.resize(x_in[i], (64, 64), interpolation=cv2.INTER_LINEAR)
        resized_input[i] = resized_image
    return resized_input

resized_x_test = resize_stft(x_test_stft)

# ==============================
# Feature extraction with encoder
# ==============================
def extract_features(saved_encoder_model, x_in):
    encoded_features = saved_encoder_model.predict(x_in)
    reshaped_features = encoded_features.reshape(encoded_features.shape[0], -1)
    return reshaped_features

# ==============================
# 2D visualization function
# ==============================
def visualize_2D_features(features_2d, labels, label_map=None,
                          title='2D Visualization', method_name="t-SNE", label_type="Ground Truth",blok = False):
    """
    Draws a 2D scatter plot for t-SNE or UMAP results.
    
    features_2d: np.array, shape (N,2)
    labels: np.array, shape (N,)
    label_map: dict {label_name -> label_id}, optional, used to generate legend
    label_type: 'Ground Truth' or 'Cluster'
    """
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # 自动生成颜色
    cmap = plt.get_cmap('tab10') if num_classes <= 10 else plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(num_classes)]

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[i]], label=(list(label_map.keys())[list(label_map.values()).index(lbl)]
                                           if label_map else f'Class {lbl}'),
                    alpha=0.7, s=15)
    
    plt.xlabel(f'{method_name} Component 1', fontsize=14)
    plt.ylabel(f'{method_name} Component 2', fontsize=14)
    plt.title(f'{method_name} 2D Visualization ({label_type})', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show(block=blok)


# ==============================
# Main process
# ==============================
saved_encoder_model = load_model("D:/vscoding/STEAD/encoder_4x4x16_v2.h5")
reshaped_features = extract_features(saved_encoder_model, resized_x_test)

# ---- t-SNE ----
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(reshaped_features)
visualize_2D_features(features_tsne, y_test, title='Ground Truth', method_name="t-SNE", label_type="Ground Truth")

# ---- UMAP ----
umap_model = umap.UMAP(n_components=2, random_state=42)
features_umap = umap_model.fit_transform(reshaped_features)
visualize_2D_features(features_umap, y_test, title='Ground Truth', method_name="UMAP", label_type="Ground Truth",blok = True)