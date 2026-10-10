import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

TITLE = "Flying Honeybee Near Flower (No Pollination Vibration)\nFull 10.0s Overview"
FULL_DURATION = 10.0
SEGMENT_DURATION = 2.5
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 256
DPI = 200
CMAP = "magma"

ROOT = Path(__file__).parent
OUT = ROOT / "频谱图输出"


def _load(p, d=None):
    y, sr = librosa.load(str(p), sr=SR, mono=True, duration=d)
    return y, sr


def _mel_db(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    t = librosa.times_like(S, sr=SR, hop_length=HOP_LENGTH)
    return S_db, t


def _draw(S_db, t, out_p, sub="", fs=(12, 5)):
    fig, ax = plt.subplots(figsize=fs, dpi=DPI)
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=SR,
                                   hop_length=HOP_LENGTH, cmap=CMAP, ax=ax)
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.set_label("dB", fontsize=10)
    title = f"{TITLE}\n{sub}" if sub else TITLE
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out_p.name}")


def _one_file(fp):
    stem = Path(fp).stem
    print(f"\n{Path(fp).name}")

    y, _ = _load(fp, FULL_DURATION)
    dur = len(y) / SR
    S_db, t = _mel_db(y)
    _draw(S_db, t, OUT / f"{stem}_全景.png", f"Full {dur:.1f}s Overview")

    starts = [0.5, dur/2 - SEGMENT_DURATION/2, dur - SEGMENT_DURATION - 0.5]
    labels_zh = ["前段细节", "中段细节", "后段细节"]
    labels_en = ["Early Segment", "Mid Segment", "Late Segment"]
    for i, (s, lzh, len_) in enumerate(zip(starts, labels_zh, labels_en)):
        s = max(0, min(s, dur - SEGMENT_DURATION))
        e = s + SEGMENT_DURATION
        y_seg = y[int(s * SR):int(e * SR)]
        S_db_s, t_s = _mel_db(y_seg)
        t_s += s
        _draw(S_db_s, t_s, OUT / f"{stem}_细节{i+1}_{lzh}.png",
              f"{len_} ({s:.1f}s - {e:.1f}s)", fs=(8, 4))


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(ROOT.glob("*.mp3")) or sorted(ROOT.glob("*.wav"))
    if not files:
        print(f"未找到音频文件: {ROOT}")
        sys.exit(1)
    print(f"找到 {len(files)} 个音频文件")
    for f in files:
        _one_file(f)
    print(f"\n完成，共 {len(files)} 个音频，输出: {OUT}")


if __name__ == "__main__":
    main()
