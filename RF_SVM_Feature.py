# RF_SVM_Feature.py
# MFCC extraction + Random Forest feature screening + Linear SVM classification
import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===================== 1. Hyperparameters (Consistent with paper) =====================
SR = 22050          # Audio sampling rate
N_MFCC = 24         # MFCC feature dimension
RF_TREE_NUM = 50    # Number of RF decision trees
SVM_C = 1.0         # SVM penalty coefficient
TEST_RATIO = 0.2    # Train-test split ratio

# ===================== 2. Extract MFCC feature from single audio =====================
def extract_mfcc(wav_path):
    y, sr = librosa.load(wav_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take average as single sample feature
    return mfcc_mean

def load_audio_dataset(root_dir):
    """
    Dataset folder structure:
    root_dir/
        pollination/   Pollination vibration audio (.wav)
        foraging/      Ordinary flower-foraging flight audio (.wav)
    """
    X = []
    y = []
    # Class 0: pollination vibration
    poll_path = os.path.join(root_dir, "pollination")
    for fname in os.listdir(poll_path):
        if fname.endswith(".wav"):
            feat = extract_mfcc(os.path.join(poll_path, fname))
            X.append(feat)
            y.append(0)
    # Class 1: ordinary foraging flight
    fly_path = os.path.join(root_dir, "foraging")
    for fname in os.listdir(fly_path):
        if fname.endswith(".wav"):
            feat = extract_mfcc(os.path.join(fly_path, fname))
            X.append(feat)
            y.append(1)
    return np.array(X), np.array(y)

# ===================== 3. RF Feature Screening (Gini coefficient) =====================
def rf_feature_select(X_train, X_test, y_train):
    rf = RandomForestClassifier(n_estimators=RF_TREE_NUM, criterion="gini", random_state=42)
    rf.fit(X_train, y_train)
    # Get feature importance score
    feat_importance = rf.feature_importances_
    # Keep top 80% important features, eliminate redundant low-contribution dimensions
    threshold = np.percentile(feat_importance, 20)
    keep_mask = feat_importance >= threshold
    X_train_filter = X_train[:, keep_mask]
    X_test_filter = X_test[:, keep_mask]
    return X_train_filter, X_test_filter, keep_mask, feat_importance

# ===================== 4. Linear SVM Training & Evaluation =====================
def train_svm(X_train, X_test, y_train, y_test):
    svm = SVC(kernel="linear", C=SVM_C)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return svm, acc

# ===================== Main Program (Simulated data, no audio folder required) =====================
if __name__ == "__main__":
    # Simulated MFCC data for code verification (No real audio files needed)
    # After organizing complete wav dataset, replace this part with load_audio_dataset()
    np.random.seed(42)
    # Simulate 200 bee audio samples, 24-dim MFCC
    X = np.random.randn(200, N_MFCC)
    # Label definition: 0 = pollination vibration, 1 = common foraging flight
    y = np.hstack([np.zeros(100), np.ones(100)])

    print(f"Total samples: {len(X)}, original MFCC dimension: {X.shape[1]}")

    # Split training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42, stratify=y)

    # Adaptive feature purification via Random Forest screening
    X_train_opt, X_test_opt, mask, imp = rf_feature_select(X_train, X_test, y_train)
    print(f"After RF screening, reserved feature dimension: {X_train_opt.shape[1]}")

    # Train lightweight linear SVM classifier
    model, test_acc = train_svm(X_train_opt, X_test_opt, y_train, y_test)
    print(f"SVM Test Accuracy = {test_acc:.4f}")

    # Output feature importance, which can be transferred to PSO weight optimization
    print("Feature importance score:\n", np.round(imp, decimals=4))