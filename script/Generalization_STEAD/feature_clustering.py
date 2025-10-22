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

# ==============================
# Data loading and preprocessing
# ==============================
x_test = np.load("x_gen_test.npy")# data for testing is generated from 'data_preprocessing.py'
y_test = np.load("y_gen_test.npy")

x_test_n = utils.norml(x_test)
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
def visualize_2D_features(features_2d, labels, title='2D Visualization', method_name="t-SNE", label_type="Ground Truth"):
    """
    Draws a 2D scatter plot for t-SNE or UMAP results.
    label_type: 'Ground Truth' or 'K-Means'
    """
    plt.figure(figsize=(7, 6))
    if label_type == "Ground Truth":
        classes = ['Earthquake', 'Noise']
        colors = ListedColormap(['blue', 'orange'])
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=colors, alpha=0.7, s=15)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    else:
        colors = ListedColormap(['green', 'red'])
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=colors, alpha=0.7, s=15)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1'])
    
    plt.xlabel(f'{method_name} Component 1', fontsize=14)
    plt.ylabel(f'{method_name} Component 2', fontsize=14)
    plt.title(f'{method_name} 2D Visualization ({label_type})', fontsize=12)
    plt.grid(True)
    plt.show(block=False)

# ==============================
# Main process
# ==============================
saved_encoder_model = load_model("model/encoder_4x4x16_v2.h5")
reshaped_features = extract_features(saved_encoder_model, resized_x_test)

# ---- t-SNE ----
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(reshaped_features)
visualize_2D_features(features_tsne, y_test, title='Ground Truth', method_name="t-SNE", label_type="Ground Truth")

# ---- UMAP ----
umap_model = umap.UMAP(n_components=2, random_state=42)
features_umap = umap_model.fit_transform(reshaped_features)
visualize_2D_features(features_umap, y_test, title='Ground Truth', method_name="UMAP", label_type="Ground Truth")

# ==============================
# K-Means clustering on 2D features
# ==============================
kmeans_tsne = KMeans(n_clusters=2, random_state=42)
kmeans_tsne.fit(features_tsne)
pred_tsne = kmeans_tsne.labels_

kmeans_umap = KMeans(n_clusters=2, random_state=42)
kmeans_umap.fit(features_umap)
pred_umap = kmeans_umap.labels_

# ---- visualize K-Means cluster labels ----
visualize_2D_features(features_tsne, pred_tsne, title='K-Means Clusters', method_name="t-SNE", label_type="K-Means")
visualize_2D_features(features_umap, pred_umap, title='K-Means Clusters', method_name="UMAP", label_type="K-Means")

# ==============================
# Confusion matrices
# ==============================
print("\n--- Confusion Matrix: t-SNE ---")
print(confusion_matrix(y_test, pred_tsne))
print(classification_report(y_test, pred_tsne, target_names=['earthquake', 'noise'], digits=4))

print("\n--- Confusion Matrix: UMAP ---")
print(confusion_matrix(y_test, pred_umap))
print(classification_report(y_test, pred_umap, target_names=['earthquake', 'noise'], digits=4))

# ==============================
# Per-class Silhouette Score
# ==============================
def per_class_silhouette(features, labels, pred_labels):
    """Compute average silhouette score per true class."""
    sil_samples = silhouette_samples(features, pred_labels)
    classes = np.unique(labels)
    for c in classes:
        class_sil = sil_samples[labels == c]
        avg_sil = np.mean(class_sil)
        name = "Earthquake" if c == 0 else "Noise"
        print(f"Average Silhouette Score for {name}: {avg_sil:.4f}")

print("\n========== Silhouette Score Summary ==========")
print("t-SNE features:")
per_class_silhouette(features_tsne, y_test, pred_tsne)

print("\nUMAP features:")
per_class_silhouette(features_umap, y_test, pred_umap)

plt.show()
