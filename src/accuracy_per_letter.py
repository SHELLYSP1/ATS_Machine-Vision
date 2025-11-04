# src/accuracy_per_letter.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# === Load hasil prediksi ===
y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

# === Buat confusion matrix ===
cm = confusion_matrix(y_true, y_pred)

# === Label huruf A-Z ===
labels = [chr(i) for i in range(65, 91)]  # ASCII 65–90 = A-Z

# === Hitung akurasi per huruf ===
print("=== AKURASI PER HURUF (A–Z) ===")
akurasi_per_huruf = []
for i, label in enumerate(labels):
    benar = cm[i, i]
    total = cm[i, :].sum()
    acc = (benar / total * 100) if total > 0 else 0
    akurasi_per_huruf.append(acc)
    print(f"Huruf {label} : {acc:.2f}%")

# === Simpan ke file teks ===
with open("accuracy_per_letter.txt", "w") as f:
    f.write("=== AKURASI PER HURUF (A–Z) ===\n")
    for label, val in zip(labels, akurasi_per_huruf):
        f.write(f"Huruf {label} : {val:.2f}%\n")

print("\n>> Hasil disimpan ke file: accuracy_per_letter.txt")

# === Visualisasi: grafik batang ===
plt.figure(figsize=(12, 6))
plt.bar(labels, akurasi_per_huruf, color="royalblue")
plt.title("Akurasi per Huruf (A–Z) - HOG + SVM", fontsize=14, fontweight='bold')
plt.xlabel("Huruf", fontsize=12)
plt.ylabel("Akurasi (%)", fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tampilkan nilai di atas batang
for i, acc in enumerate(akurasi_per_huruf):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("accuracy_per_letter_chart.png", dpi=300)
plt.show()
