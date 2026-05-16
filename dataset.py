"""
Multimodal Dataset Loader — DeepSense 6G Scenario 9
====================================================
4 client modalities (NO mmWave as input):
  Client 0 : Radar     (unit1_radar)
  Client 1 : Camera    (unit1_rgb)
  Client 2 : LiDAR     (unit1_lidar)
  Client 3 : GPS Cal.  (unit2_loc_cal)

mmWave loaded separately for TEST COMPARISON ONLY.
"""

import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from PIL import Image
import torch

def _fix(root, rel_path):
    rel_path = re.sub(r'^[./\\]+', '', str(rel_path))
    return os.path.join(root, rel_path)

# ── Client 0 — Radar ──────────────────────────────────────
def load_radar(root, csv_path, label_col='unit1_beam_index',
               col='unit1_radar', fft_size=64, max_samples=None):
    df = pd.read_csv(csv_path).dropna(subset=[col, label_col])
    if max_samples: df = df.iloc[:max_samples]
    samples, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Radar'):
        path = _fix(root, row[col])
        if not os.path.exists(path): continue
        raw = loadmat(path)['data'].astype(np.complex64)
        d = torch.from_numpy(raw)
        d = torch.fft.fft(d, dim=1)
        d -= d.mean(dim=2, keepdim=True)
        d = torch.fft.fft(d, n=fft_size, dim=0)
        feat = torch.abs(d).sum(dim=2).numpy().T
        samples.append(feat[np.newaxis,:,:].astype(np.float32))
        labels.append(int(row[label_col]))
    return np.stack(samples), np.array(labels, dtype=np.int64)

# ── Client 1 — Camera ─────────────────────────────────────
def load_camera(root, csv_path, label_col='unit1_beam_index',
                col='unit1_rgb', img_size=128, max_samples=None):
    df = pd.read_csv(csv_path).dropna(subset=[col, label_col])
    if max_samples: df = df.iloc[:max_samples]
    samples, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Camera'):
        path = _fix(root, row[col])
        if not os.path.exists(path): continue
        img = Image.open(path).convert('RGB').resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        samples.append(arr.transpose(2,0,1))
        labels.append(int(row[label_col]))
    return np.stack(samples), np.array(labels, dtype=np.int64)

# ── Client 2 — LiDAR ──────────────────────────────────────
def load_lidar(root, csv_path, label_col='unit1_beam_index',
               col='unit1_lidar', max_pts=512, max_samples=None):
    df = pd.read_csv(csv_path).dropna(subset=[col, label_col])
    if max_samples: df = df.iloc[:max_samples]
    samples, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='LiDAR'):
        path = _fix(root, row[col])
        if not os.path.exists(path): continue
        mat = loadmat(path)
        key = [k for k in mat if not k.startswith('_')][0]
        pts = mat[key].astype(np.float32)
        if pts.ndim == 1: pts = pts.reshape(-1,1)
        if pts.shape[0] < pts.shape[1]: pts = pts.T
        N = pts.shape[0]
        if N >= max_pts:
            pts = pts[:max_pts]
        else:
            pts = np.vstack([pts, np.zeros((max_pts-N, pts.shape[1]), dtype=np.float32)])
        samples.append(pts.flatten())
        labels.append(int(row[label_col]))
    return np.stack(samples), np.array(labels, dtype=np.int64)

# ── Client 3 — GPS Calibrated ─────────────────────────────
def load_gps(root, csv_path, label_col='unit1_beam_index',
             col='unit2_loc_cal', max_samples=None):
    df = pd.read_csv(csv_path).dropna(subset=[col, label_col])
    if max_samples: df = df.iloc[:max_samples]
    samples, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='GPS'):
        path = _fix(root, row[col])
        if not os.path.exists(path): continue
        coord = np.loadtxt(path, dtype=np.float32)
        samples.append(coord[:2])
        labels.append(int(row[label_col]))
    X = np.stack(samples)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, np.array(labels, dtype=np.int64)

# ── mmWave — FOR TESTING/COMPARISON ONLY (not a client) ───
def load_mmwave_test(root, csv_path, label_col='unit1_beam_index',
                     col='unit1_pwr_60ghz', max_samples=None):
    """
    Loads mmWave 64-beam power vectors.
    Used ONLY to compare our FL prediction vs mmWave baseline.
    NOT used as training input.
    """
    df = pd.read_csv(csv_path).dropna(subset=[col, label_col])
    if max_samples: df = df.iloc[:max_samples]
    samples, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='mmWave(test)'):
        path = _fix(root, row[col])
        if not os.path.exists(path): continue
        samples.append(np.loadtxt(path, dtype=np.float32))
        labels.append(int(row[label_col]))
    return np.stack(samples), np.array(labels, dtype=np.int64)

# ── Public loader dict (4 clients only) ───────────────────
LOADERS = {
    0: load_radar,
    1: load_camera,
    2: load_lidar,
    3: load_gps,
}