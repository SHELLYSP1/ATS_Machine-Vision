# ==========================================================
# Program : Klasifikasi Huruf Tulisan Tangan (EMNIST Letters)
# Metode  : HOG + SVM
# Evaluasi: Leave-One-Out Cross Validation (LOOCV)
# ==========================================================

import os
import time
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report
from skimage.feature import hog
import cv2

# ========== KONFIGURASI ==========
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "emnist-letters-train.csv")
SAMPLES_PER_CLASS = 500    # 500 x 26 total 13000
RANDOM_STATE = 42

# ========== FUNGSI UTILITAS ==========
def load_and_balance(csv_path, samples_per_class=500):
    print(f"📂 Loading dataset dari: {csv_path}")
    df = pd.read_csv(csv_path)

    # Kolom pertama = label, sisanya = pixel (784 kolom)
    balanced_df = []
    for label in sorted(df.iloc[:, 0].unique()):
        subset = df[df.iloc[:, 0] == label]
        if len(subset) < samples_per_class:
            print(f"⚠️  Class {label} hanya memiliki {len(subset)} data, akan diambil semua.")
            sampled = subset
        else:
            sampled = resample(subset, n_samples=samples_per_class, random_state=RANDOM_STATE)
        balanced_df.append(sampled)

    df_balanced = pd.concat(balanced_df).reset_index(drop=True)
    y = df_balanced.iloc[:, 0].values
    X = df_balanced.iloc[:, 1:].values
    print(f"✅ Data seimbang dibuat: {len(y)} total sampel, {len(np.unique(y))} kelas.\n")
    return X, y


def preprocess_image_vector(vec):
    """Ubah 1D vector (28x28) jadi citra 2D dan koreksi orientasi (tanpa flip)"""
    img = np.reshape(vec, (28, 28)).astype(np.uint8)
    # hanya transpose agar huruf tegak, tanpa pembalikan
    img = img.T
    return img


def extract_hog_features(X):
    print("⚙️  Ekstraksi fitur HOG dimulai ...")
    features = []
    start = time.time()
    for i, vec in enumerate(X):
        img = preprocess_image_vector(vec)
        feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        features.append(feat)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(X)} gambar diproses...")
    print(f"✅ Ekstraksi HOG selesai dalam {time.time() - start:.2f} detik.\n")
    return np.array(features)


# ========== MAIN PROGRAM ==========
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ File dataset tidak ditemukan di {DATA_PATH}")

    # Load dan sampling seimbang
    X_raw, y = load_and_balance(DATA_PATH, samples_per_class=SAMPLES_PER_CLASS)

    # Simpan data mentah
    np.save("X_raw.npy", X_raw)
    print("💾 File X_raw.npy berhasil disimpan (data citra mentah untuk visualisasi).")

    # Ekstraksi fitur HOG
    hog_features = extract_hog_features(X_raw)

    # Model SVM
    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)

    # Leave-One-Out Cross Validation
    print("🚀 Menjalankan Leave-One-Out Cross Validation (LOOCV)...")
    cv = LeaveOneOut()
    print(f"📈 Jumlah iterasi: {cv.get_n_splits(X_raw)}")

    start = time.time()
    y_pred = cross_val_predict(clf, hog_features, y, cv=cv, n_jobs=-1)
    print(f"✅ Proses LOOCV selesai dalam {time.time() - start:.2f} detik.\n")

    # Evaluasi performa
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print("\n📊 HASIL EVALUASI (LOOCV)")
    print("="*40)
    print(f"Akurasi   : {acc*100:.2f}%")
    print(f"Presisi   : {prec*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print("="*40)
    print("\nLaporan klasifikasi:\n", classification_report(y, y_pred, zero_division=0))

    # Simpan hasil
    np.save("confusion_matrix.npy", cm)
    np.save("y_true.npy", y)
    np.save("y_pred.npy", y_pred)
    print("💾 confusion_matrix.npy, y_true.npy, dan y_pred.npy telah disimpan.")

    print("\n✅ Program selesai tanpa error.")


if __name__ == "__main__":
    main()
