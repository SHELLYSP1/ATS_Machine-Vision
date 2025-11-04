# src/plot_cm_full.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

# Load hasil evaluasi
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

# Hitung metrik
acc = accuracy_score(y_true, y_pred) * 100
prec = precision_score(y_true, y_pred, average='macro') * 100
f1 = f1_score(y_true, y_pred, average='macro') * 100

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Buat label huruf A-Z
labels = [chr(i) for i in range(65, 91)]  # 65-90 = A-Z

# === CETAK METRIK DI TERMINAL ===
print("=== HASIL EVALUASI LOOCV (IMPROVED) ===")
print(f"Akurasi : {acc:.2f}%")
print(f"Presisi : {prec:.2f}%")
print(f"F1-Score: {f1:.2f}%")

# === PLOTTING ===
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=True)

plt.title("Confusion Matrix - LOOCV (EMNIST Letters, Improved)")
plt.xlabel("Predicted label")
plt.ylabel("True label")

# Tambahkan teks hasil metrik di atas grafik
plt.figtext(0.1, 0.95, "=== HASIL EVALUASI LOOCV (IMPROVED) ===", fontsize=10, fontweight='bold')
plt.figtext(0.1, 0.93, f"Akurasi : {acc:.2f}%", fontsize=9)
plt.figtext(0.1, 0.91, f"Presisi : {prec:.2f}%", fontsize=9)
plt.figtext(0.1, 0.89, f"F1-Score: {f1:.2f}%", fontsize=9)

# Simpan hasil
plt.savefig("confusion_matrix_full.png", dpi=300, bbox_inches='tight')
plt.show()
