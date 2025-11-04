# src/plot_cm.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load hasil evaluasi
cm = np.load("confusion_matrix.npy")
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

# Tampilkan ringkasan data
print("Confusion matrix shape:", cm.shape)
print("Total samples:", len(y_true))

# Plot confusion matrix pakai heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap='Blues', cbar=True)
plt.title("Confusion Matrix - HOG + SVM (EMNIST Letters)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
