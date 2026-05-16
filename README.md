# Multimodal Heterogeneous Federated Learning for 6G mmWave Beam Prediction

## About This Project
In this project, we worked on predicting the best mmWave communication beam using multiple types of sensor data - radar, camera, LiDAR, and GPS. The main idea is that in real 6G networks, different devices or units might have different sensors. We wanted to let them train together without sharing their raw data, which is where federated learning comes in.

Instead of one device having all the data, we have 4 clients, each with a different sensor type. They train locally and only send the weights of a small shared network to the server. The server combines these weights using FedAvg and sends them back. No raw data ever leaves the client.

We also added an attention fusion step after the main training. This helps the system figure out which sensor is more useful for each sample - for example, camera turns out to be the most reliable (48.9% weight) while LiDAR is the least useful (4.8%).



## How the System Works

We split training into two phases.

**Phase 1 - Federated Training (30 rounds)**

Each of the 4 clients trains on their own sensor data for 3 local epochs. After that they send only the shared prediction head weights (around 246 KB) to the server. The server averages them using FedAvg and sends the updated head back. This repeats for 30 rounds.

```
Client 0 (Radar)  ─┐
Client 1 (Camera) ─┤──► Server (FedAvg) ──► updated shared head ──► repeat
Client 2 (LiDAR)  ─┤    only 246 KB sent per client per round
Client 3 (GPS)    ─┘
```

**Phase 2 - Attention Fine-Tuning (10 epochs)**

After Phase 1, the encoders and head are frozen. Each client sends only their 128-dimensional embeddings (not raw data) to the server. The server trains a small attention network that learns how much to trust each modality for each sample. This gives dynamic weights per sample rather than fixed weights.

```
4 client embeddings ──► Attention network ──► weighted fused embedding ──► beam prediction
```

---

## Results

We tested on the full DeepSense 6G Scenario 9 dataset with 5,964 samples.

**Phase 1 best validation accuracy**

| Metric | Result |
|--------|--------|
| Top-1  | 50.39% |
| Top-5  | 94.53% |

**Phase 2 best validation accuracy (after attention fine-tuning)**

| Metric | Result |
|--------|--------|
| Top-1  | 56.09% |
| Top-5  | 97.65% |

**Final test accuracy for different sensor combinations**

| Combination | Top-1 | Top-2 | Top-3 | Top-5 |
|-------------|-------|-------|-------|-------|
| All 4 sensors (attention fusion) | 55.98% | 78.10% | 89.50% | 96.98% |
| Radar + Camera | 55.75% | 77.65% | 88.94% | 96.98% |
| Camera + GPS | 54.41% | 76.65% | 88.16% | 96.87% |
| Camera only | 54.53% | 76.20% | 88.16% | 96.98% |
| Radar only | 45.59% | 65.81% | 79.33% | 93.74% |
| GPS only | 29.72% | 46.37% | 56.42% | 72.40% |
| LiDAR only | 27.60% | 42.68% | 52.51% | 69.83% |
| mmWave baseline | - | 43.80% | 74.41% | 93.74% |

**What attention weights the network learned**

| Sensor | Average weight |
|--------|---------------|
| Camera | 48.9% |
| Radar  | 32.2% |
| GPS    | 14.1% |
| LiDAR  | 4.8%  |

Camera is most reliable, which makes sense since you can visually see where a vehicle is. LiDAR gets very low weight because our simple MLP on flat point clouds does not extract 3D structure well.

The beam label comes from the mmWave argmax.
---

## Files in This Repo

```
train.py      main training script - Phase 1 FL + Phase 2 attention
dataset.py    loads radar, camera, LiDAR, GPS, and mmWave data from CSV
models.py     encoder networks for each modality + attention fusion + prediction head
README.md     
```

---

## Model Details

**Encoders - each client keeps these private**

| Client | Sensor | Model | Output size |
|--------|--------|-------|-------------|
| 0 | Radar  | 3-layer CNN with BatchNorm | 128-dim |
| 1 | Camera | 4-layer CNN with BatchNorm | 128-dim |
| 2 | LiDAR  | 3-layer MLP on flat point cloud | 128-dim |
| 3 | GPS    | 3-layer MLP on lat/lon coordinates | 128-dim |

All encoders output the same 128-dimensional vector so they can be fused.

**Shared prediction head - this is what gets averaged by FedAvg**

```
128-dim → Linear(256) → ReLU → Dropout
       → Linear(128) → ReLU → Dropout
       → Linear(64)  → 64 beam logits → argmax → predicted beam
```

**Attention fusion network - trained in Phase 2**

```
For each embedding: Linear(128→64) → Tanh → Linear(64→1) → score
Softmax over scores → per-sample weights → weighted average → 128-dim
```

---

## Training Settings

| Setting | Value |
|---------|-------|
| FL rounds | 30 |
| Local epochs per round | 3 |
| Attention fine-tuning epochs | 10 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Embedding size | 128 |
| Number of beams | 64 |
| Loss function | Cross-entropy |
| Total samples | 5,964 |
| Split | 70% train / 15% val / 15% test |

---

## Dataset

We used **DeepSense 6G Scenario 9**, collected at Arizona State University. Vehicles drive past a roadside base station that records synchronized sensor data and mmWave beam measurements.

Download from: https://www.deepsense6g.net

After downloading, the folder should look like this:

```
scenario9_dev/
    scenario9.csv
    unit1/
        radar_data/
        image_data/
        lidar_data/
        power_60GHz/
    unit2/
        GPS_data/
```

---

## How to Run

**1. Install required packages**

```bash
pip install torch torchvision numpy pandas scipy Pillow tqdm scikit-learn matplotlib
```

**2. Set your dataset path in train.py**

```python
ROOT_DIR = '/path/to/scenario9_dev'
CSV_FILE = '/path/to/scenario9_dev/scenario9.csv'
```

**3. Run**

```bash
python train.py






