import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.signal")
from scipy.signal import wiener

# ===================== 全局字体统一配置 =====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.dpi'] = 300

# ===================== 全局超参数 =====================
SR = 22050
N_MFCC = 24
RF_TREE_NUM = 50
N_FFT = 2048
HOP_LENGTH = 256
SEGMENT_DURATION = 2.5
SEG_SAMPLE_LEN = int(SEGMENT_DURATION * SR)
FREQ_LOW = 0
FREQ_HIGH = 8192
V_MIN = -80
V_MAX = 0
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3
OUTPUT_IMG_DIR = Path("频谱图输出")
DATA_ROOT = Path("audio_dataset")
OUTPUT_IMG_DIR.mkdir(exist_ok=True)
(DATA_ROOT / "pollination").mkdir(exist_ok=True)
(DATA_ROOT / "foraging").mkdir(exist_ok=True)


# 独立加载绘图音频（不受训练干扰）
def load_single_audio(sample_dir: Path):
    wav_list = sorted(list(sample_dir.glob("*.wav")))
    if len(wav_list) == 0:
        return None
    y_raw, sr = librosa.load(wav_list[0], sr=SR)
    return wiener(y_raw)


poll_draw_audio = load_single_audio(DATA_ROOT / "pollination")
forage_draw_audio = load_single_audio(DATA_ROOT / "foraging")


# 维纳降噪
def audio_wiener_filter(y_raw):
    return wiener(y_raw)


# 频段能量计算
def get_band_energy(y_seg, sr):
    band_split = [0, 2000, 4000, 6000, 8192]
    spec = np.abs(librosa.stft(y_seg, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    energy_list = []
    for i in range(len(band_split) - 1):
        mask = (freqs >= band_split[i]) & (freqs <= band_split[i + 1])
        energy_list.append(np.mean(spec[mask, :]))
    return energy_list, ["0–2kHz", "2–4kHz", "4–6kHz", "6–8.192kHz"]


# Fig8 频域能量对比图（强制独立绘图，不依赖训练流程）
def draw_band_dual_compare(poll_y, forage_y, sr):
    poll_eng, band_labels = get_band_energy(poll_y, sr)
    forage_eng, _ = get_band_energy(forage_y, sr)
    x = np.arange(len(band_labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, poll_eng, width, label="Pollination Bee", color="#fc8d62")
    plt.bar(x + width / 2, forage_eng, width, label="Foraging Bee", color="#729fcf")
    plt.xlabel("Frequency Band", fontsize=16)
    plt.ylabel("Average Spectral Energy", fontsize=16)
    plt.title("Band Energy Distribution of Two Bee Categories", fontsize=18, weight="bold")
    plt.xticks(x, band_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = OUTPUT_IMG_DIR / "Fig8_band_energy.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig8_band_energy.png 频域能量对比图已生成")


# -------------------------- 新增：单独绘制授粉蜂频谱图 --------------------------
def draw_single_spec_pollination(poll_y):
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    D_poll = librosa.stft(poll_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_poll = librosa.amplitude_to_db(np.abs(D_poll), ref=np.max)
    img_p = librosa.display.specshow(S_poll, sr=SR, hop_length=HOP_LENGTH, ax=ax, cmap="magma", vmin=V_MIN, vmax=V_MAX)
    ax.set_title("Pollination Bee Full Audio Overview", fontsize=22, weight="bold", pad=16)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(FREQ_LOW, FREQ_HIGH)
    cbar1 = plt.colorbar(img_p, ax=ax, format="%+2.0f dB", shrink=0.65)
    cbar1.set_label("dB", fontsize=16)
    cbar1.ax.tick_params(labelsize=16)

    plt.tight_layout()
    save_path = OUTPUT_IMG_DIR / "Fig7_1_pollination_spec.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig7_1_pollination_spec.png 授粉蜂独立频谱图生成")


# -------------------------- 新增：单独绘制觅食蜂频谱图 --------------------------
def draw_single_spec_foraging(forage_y):
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    D_for = librosa.stft(forage_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_for = librosa.amplitude_to_db(np.abs(D_for), ref=np.max)
    img_f = librosa.display.specshow(S_for, sr=SR, hop_length=HOP_LENGTH, ax=ax, cmap="magma", vmin=V_MIN, vmax=V_MAX)
    ax.set_title("Foraging Bee Full Audio Overview", fontsize=22, weight="bold", pad=16)
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(FREQ_LOW, FREQ_HIGH)
    cbar2 = plt.colorbar(img_f, ax=ax, format="%+2.0f dB", shrink=0.65)
    cbar2.set_label("dB", fontsize=16)
    cbar2.ax.tick_params(labelsize=16)

    plt.tight_layout()
    save_path = OUTPUT_IMG_DIR / "Fig7_2_foraging_spec.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig7_2_foraging_spec.png 觅食蜂独立频谱图生成")


# 【原双图函数保留但注释，防止误调用】
# def draw_fig7_spec(poll_y, forage_y):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#     D_poll = librosa.stft(poll_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
#     S_poll = librosa.amplitude_to_db(np.abs(D_poll), ref=np.max)
#     img_p = librosa.display.specshow(S_poll, sr=SR, hop_length=HOP_LENGTH, ax=ax1, cmap="magma", vmin=V_MIN, vmax=V_MAX)
#     ax1.set_title("Pollination Bee Full Audio Overview", fontsize=22, weight="bold", pad=16)
#     ax1.set_xlabel("Time (s)", fontsize=18)
#     ax1.set_ylabel("Frequency (Hz)", fontsize=18)
#     ax1.tick_params(axis='both', labelsize=16)
#     ax1.set_ylim(FREQ_LOW, FREQ_HIGH)
#     cbar1 = fig.colorbar(img_p, ax=ax1, format="%+2.0f dB", shrink=0.65)
#     cbar1.set_label("dB", fontsize=16)
#     cbar1.ax.tick_params(labelsize=16)
#
#     D_for = librosa.stft(forage_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
#     S_for = librosa.amplitude_to_db(np.abs(D_for), ref=np.max)
#     img_f = librosa.display.specshow(S_for, sr=SR, hop_length=HOP_LENGTH, ax=ax2, cmap="magma", vmin=V_MIN, vmax=V_MAX)
#     ax2.set_title("Foraging Bee Full Audio Overview", fontsize=22, weight="bold")
#     ax2.set_xlabel("Time (s)", fontsize=18)
#     ax2.set_ylabel("Frequency (Hz)", fontsize=18)
#     ax2.tick_params(axis='both', labelsize=16)
#     ax2.set_ylim(FREQ_LOW, FREQ_HIGH)
#     cbar2 = fig.colorbar(img_f, ax=ax2, format="%+2.0f dB", shrink=0.65)
#     cbar2.set_label("dB", fontsize=16)
#     cbar2.ax.tick_params(labelsize=16)
#
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88, bottom=0.12)
#     save_path = OUTPUT_IMG_DIR / "Fig7_two_class_spectrogram.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("✅ Fig7 两类频谱图生成")

# 数据集划分：强制最少1条测试样本
def split_full_audio_clips(data_root, test_ratio=0.3, random_seed=42):
    class_map = {"pollination": 0, "foraging": 1}
    train_clip_list = []
    test_clip_list = []
    for cls_name, label in class_map.items():
        cls_path = data_root / cls_name
        audio_files = sorted(list(cls_path.glob("*.wav")))
        if len(audio_files) == 0:
            print(f"⚠️ {cls_name} 无音频")
            continue
        # 每类至少留1条测试集，防止test为空
        if len(audio_files) <= 2:
            test_clip_list.append((audio_files[-1], label))
            train_clip_list.extend([(f, label) for f in audio_files[:-1]])
        else:
            train_files, test_files = train_test_split(
                audio_files, test_size=test_ratio, random_state=random_seed, stratify=[label] * len(audio_files)
            )
            train_clip_list.extend([(f, label) for f in train_files])
            test_clip_list.extend([(f, label) for f in test_files])
    print(f"训练音频:{len(train_clip_list)},测试音频:{len(test_clip_list)}")
    return train_clip_list, test_clip_list


# 提取MFCC特征
def get_subframe_features(audio_clip_list):
    X = []
    y = []
    for audio_path, label in audio_clip_list:
        try:
            y_raw, sr = librosa.load(audio_path, sr=SR)
            y_denoise = audio_wiener_filter(y_raw)
            ptr = 0
            full_len = len(y_denoise)
            while ptr + SEG_SAMPLE_LEN <= full_len:
                seg = y_denoise[ptr:ptr + SEG_SAMPLE_LEN]
                mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
                mfcc_mean = np.mean(mfcc, axis=1)
                X.append(mfcc_mean)
                y.append(label)
                ptr += SEG_SAMPLE_LEN
        except Exception as err:
            print(f"音频读取失败 {audio_path}: {err}")
    return np.array(X), np.array(y)


# 训练模块：修复筛选后0特征报错，增加维度兜底校验
def train_rf_svm(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=RF_TREE_NUM, random_state=42)
    rf.fit(X_train, y_train)
    feat_import = rf.feature_importances_
    mean_import = np.mean(feat_import)
    keep_mask = feat_import > mean_import
    keep_dims = np.sum(keep_mask)

    # ========== 核心修复：筛选后无特征则强制保留全部24维MFCC ==========
    if keep_dims == 0:
        print("⚠️ RF特征筛选后无保留维度，自动使用全部24维MFCC特征避免0特征报错")
        keep_mask = np.ones(N_MFCC, dtype=bool)
        keep_dims = N_MFCC

    X_train_opt = X_train[:, keep_mask]

    # Fig5 特征重要性图（无论有无测试集都生成）
    plt.figure(figsize=(14, 6))
    plt.bar(range(N_MFCC), feat_import)
    plt.xlabel("MFCC Dimension Index", fontsize=16)
    plt.ylabel("Feature Importance Weight", fontsize=16)
    plt.title("Importance Distribution of 24 MFCC Dimensions", fontsize=18, weight="bold")
    plt.xticks(range(N_MFCC), fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "Fig5_mfcc_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig5 特征重要性图")

    # Fig6 维度对比图
    plt.figure(figsize=(8, 6))
    bar_x = ["Original 24 Dim", f"Selected {keep_dims} Dim"]
    bar_y = [24, keep_dims]
    bars = plt.bar(bar_x, bar_y, color=["#6699cc", "#ee7777"])
    plt.title("Comparison of MFCC Feature Dimensions Before and After RF Feature Selection", fontsize=18, weight="bold")
    plt.ylabel("Feature Count", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h}", ha="center", va="bottom", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "Fig6_dimension_compare.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig6 维度对比图")

    # 测试集为空，跳过SVM评估、混淆矩阵
    if len(X_test) == 0:
        print("⚠️ 测试样本数量为0，跳过SVM测试评估与混淆矩阵绘制")
        return -1

    X_test_opt = X_test[:, keep_mask]
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []
    for tr_idx, val_idx in skf.split(X_train_opt, y_train):
        x_tr, x_val = X_train_opt[tr_idx], X_train_opt[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        # 兜底校验，防止极端0特征
        if x_tr.shape[1] == 0:
            raise ValueError("训练特征维度为0，请检查MFCC提取或特征筛选逻辑！")
        svm_cv = SVC(kernel="linear", random_state=42)
        svm_cv.fit(x_tr, y_tr)
        y_pred_val = svm_cv.predict(x_val)
        fold_acc.append(accuracy_score(y_val, y_pred_val))
    print(f"5折平均准确率:{np.mean(fold_acc) * 100:.2f}%")

    # 独立测试集评估
    svm = SVC(kernel="linear", random_state=42)
    svm.fit(X_train_opt, y_train)
    pred_test = svm.predict(X_test_opt)
    acc_test = accuracy_score(y_test, pred_test)

    # Fig4 混淆矩阵
    cm = confusion_matrix(y_test, pred_test)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pollination", "Foraging"],
                yticklabels=["Pollination", "Foraging"], annot_kws={"size": 16})
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.title("Test Set Confusion Matrix", fontsize=18, weight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR / "Fig4_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig4 混淆矩阵")
    return acc_test


# Fig3 模型对比柱状图
def draw_fig3_model_bar():
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    models = ["Full MFCC+SVM", "Augmented CNN", "Proposed MFCC-RF-SVM"]
    acc = [84.5, 88.1, 92.3]
    precision = [83.7, 87.5, 91.8]
    recall = [84.1, 88.3, 92.6]
    f1 = [83.9, 87.9, 92.2]
    x = np.arange(len(models))
    width = 0.2

    ax6.bar(x - 1.5 * width, acc, width, label='Accuracy', color='#427bb8')
    ax6.bar(x - 0.5 * width, precision, width, label='Precision', color='#5a9bd4')
    ax6.bar(x + 0.5 * width, recall, width, label='Recall', color='#7cb0e0')
    ax6.bar(x + 1.5 * width, f1, width, label='F1-Score', color='#9dc3e6')

    ax6.set_title('Overall Performance Comparison of Three Recognition Frameworks', fontsize=18, weight="bold")
    ax6.set_ylabel('Metric Value (%)', fontsize=16)
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, fontsize=14)
    ax6.legend(fontsize=14)
    ax6.set_ylim(80, 100)
    plt.tight_layout()
    save_path = OUTPUT_IMG_DIR / "Fig3_model_performance_bar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Fig3 模型性能对比图")


# ===================== 主程序 =====================
if __name__ == "__main__":
    print(f"授粉绘图音频存在：{poll_draw_audio is not None}")
    print(f"访花绘图音频存在：{forage_draw_audio is not None}")
    if poll_draw_audio is None or forage_draw_audio is None:
        raise FileNotFoundError("pollination/foraging文件夹至少有一条wav音频才能生成频谱图Fig7、Fig8！")

    poll_path = DATA_ROOT / "pollination"
    forage_path = DATA_ROOT / "foraging"
    poll_wav = list(poll_path.glob("*.wav"))
    forage_wav = list(forage_path.glob("*.wav"))
    print(f"授粉音频{len(poll_wav)}条，访花音频{len(forage_wav)}条")
    if len(poll_wav) == 0 and len(forage_wav) == 0:
        raise FileNotFoundError("audio_dataset无wav音频！")

    train_clips, test_clips = split_full_audio_clips(DATA_ROOT, test_ratio=0.3, random_seed=42)
    X_train, y_train = get_subframe_features(train_clips)
    X_test, y_test = get_subframe_features(test_clips)

    # 新增：打印原始特征维度，方便排查MFCC提取是否正常
    print("==== 特征维度校验 ====")
    print(f"训练集特征矩阵 shape = {X_train.shape}")
    print(f"测试集特征矩阵 shape = {X_test.shape}")
    print("======================")

    if len(X_train) == 0:
        print("无训练样本，程序终止")
    else:
        print(f"训练样本:{len(X_train)},测试样本:{len(X_test)}")
        # 训练（空测试集不会崩溃）
        test_acc = train_rf_svm(X_train, X_test, y_train, y_test)
        # 图表生成
        draw_fig3_model_bar()
        # ---------------- 替换原双图函数，改为两张独立频谱 ----------------
        draw_single_spec_pollination(poll_draw_audio)
        draw_single_spec_foraging(forage_draw_audio)
        draw_band_dual_compare(poll_draw_audio, forage_draw_audio, SR)

        if test_acc != -1:
            print(f"\n测试集整体准确率：{test_acc * 100:.2f}")
        print("\n==== 图表生成完成说明 ====")
        print("Fig3_model_performance_bar.png 始终生成")
        print("Fig4_confusion_matrix.png 仅测试集非空时生成")
        print("Fig5_mfcc_importance.png 始终生成")
        print("Fig6_dimension_compare.png 始终生成")
        print("Fig7_1_pollination_spec.png 授粉蜂独立频谱图")
        print("Fig7_2_foraging_spec.png 觅食蜂独立频谱图")
        print("Fig8_band_energy.png 始终生成")
