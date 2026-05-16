"""
models.py — Multimodal FL with Attention Fusion

4 client encoders + Shared Head + Attention Fusion Network
- Encoders  : private per client (never shared)
- Head      : aggregated by FedAvg
- Attention : aggregated by FedAvg (trained at server)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM   = 128
NUM_CLASSES = 64


#  Attention Fusion Network 
# Takes N embeddings - computes per-sample dynamic weights
# - weighted sum - single fused embedding
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        # Scorer: embedding - scalar attention score
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, embeddings):
        """
        embeddings: list of tensors each (B, embed_dim)
        returns   : fused tensor (B, embed_dim)
        """
        # Score each embedding
        scores = torch.stack(
            [self.scorer(e) for e in embeddings], dim=1
        )  # (B, N, 1)

        # Dynamic per-sample weights via softmax
        weights = F.softmax(scores, dim=1)  # (B, N, 1)

        # Weighted sum
        stacked = torch.stack(embeddings, dim=1)  # (B, N, embed_dim)
        fused   = (weights * stacked).sum(dim=1)  # (B, embed_dim)
        return fused, weights.squeeze(-1)          # also return weights for analysis


#  Shared Prediction Head 
class PredictionHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


#  Client 0 — Radar CNN 
class RadarEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().reshape(x.size(0), -1)
        return self.fc(x)


# ─ Client 1 — Camera CNN 
class CameraEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,  16, 3, padding=1), nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().reshape(x.size(0), -1)
        return self.fc(x)


# ─ Client 2 — LiDAR MLP (lazy) 
class LidarEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = None

    def _build(self, flat_dim, device):
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, self.embed_dim),
        ).to(device)

    def forward(self, x):
        x = x.contiguous().reshape(x.size(0), -1)
        if self.net is None:
            self._build(x.shape[1], x.device)
        return self.net(x)


# ─ Client 3 — GPS MLP 
class GpsEncoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),       nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
    def forward(self, x):
        return self.net(x)


# ─ Full Client Model (Encoder + Head)
class ClientModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head    = head

    def forward(self, x):
        return self.head(self.encoder(x))


# ─ Factory 
ENCODERS = {
    0: RadarEncoder,
    1: CameraEncoder,
    2: LidarEncoder,
    3: GpsEncoder,
}
