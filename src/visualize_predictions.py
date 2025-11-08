# ==========================================================
# Program : Visualisasi Hasil Prediksi Huruf (A–Z)
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
X = np.load("X_raw.npy")
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

# === Label huruf A-Z ===
labels = [chr(i) for i in range(65, 91)]  # 65–90 = A–Z

# === Ambil contoh per huruf (pakai median agar tidak acak) ===
indices = []
for label_num in range(1, 27):  # label 1–26 = huruf A–Z
    idx = np.where(y_true == label_num)[0]
    if len(idx) > 0:
        mid = len(idx) // 2  # ambil data di tengah
        indices.append(idx[mid])
indices = np.array(indices)

print(f"Menampilkan {len(indices)} huruf berurutan A–Z.")

# === Plot grid ===
plt.figure(figsize=(15, 6))
for i, idx in enumerate(indices):
    img = X[idx].reshape(28, 28).astype("uint8")
    img = img.T  # orientasi tegak tanpa mirror

    plt.subplot(3, 9, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Pastikan label sesuai 1–26
    true_label = labels[int(y_true[idx]) - 1 if int(y_true[idx]) > 0 else 0]
    pred_label = labels[int(y_pred[idx]) - 1 if int(y_pred[idx]) > 0 else 0]

    plt.title(f"Pred: {pred_label}", fontsize=9, pad=2)
    plt.xlabel(f"True: {true_label}", fontsize=8)

plt.suptitle("Prediksi Huruf Tulisan Tangan (Urutan A–Z)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(pad=1.2)
plt.subplots_adjust(top=0.9)
plt.savefig("visualize_predictions_ordered_fixed2.png", dpi=300, bbox_inches='tight')
plt.show()
