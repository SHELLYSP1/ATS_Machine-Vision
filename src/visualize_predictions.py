# src/visualize_predictions_ordered.py
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
X = np.load("X_raw.npy")
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

# === Label huruf A-Z ===
labels = [chr(i) for i in range(65, 91)]  # 65–90 = A-Z

# === Ambil 1 contoh per huruf ===
indices = []
for label_num in range(1, 27):  # label 1-26 = huruf A-Z
    idx = np.where(y_true == label_num)[0]
    if len(idx) > 0:
        indices.append(idx[0])  # ambil 1 contoh pertama
indices = np.array(indices)

print(f"Menampilkan {len(indices)} huruf berurutan A–Z.")

# === Plot grid ===
plt.figure(figsize=(15, 6))
for i, idx in enumerate(indices):
    img = X[idx].reshape(28, 28).astype("uint8")

    # Perbaiki orientasi EMNIST
    img = np.flipud(img.T)

    plt.subplot(3, 9, i + 1)  # 3 baris × 9 kolom (27 slot cukup)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    pred_label = labels[int(y_pred[idx]) - 1]
    true_label = labels[int(y_true[idx]) - 1]

    # Tampilkan prediksi & label
    plt.title(f"Pred: {pred_label}", fontsize=9, pad=2)
    plt.xlabel(f"True: {true_label}", fontsize=8)

plt.suptitle("Prediksi Huruf Tulisan Tangan (Urutan A–Z)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(pad=1.2)
plt.subplots_adjust(top=0.9)
plt.savefig("visualize_predictions_ordered.png", dpi=300, bbox_inches='tight')
plt.show()
