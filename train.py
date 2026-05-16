"""
Attention Fusion Multimodal FL
================================
Phase 1: Standard FL training (30 rounds) — each client trains locally
Phase 2: Server-side attention fine-tuning (10 epochs)
         — server collects embeddings from all clients (not raw data)
         — trains AttentionFusion network on those embeddings
         — result: dynamic per-sample weights → ALL > subset > single

Expected order:
  ALL (Attention) > Radar+Camera > Camera+GPS > Camera > Radar > GPS > LiDAR
"""

import os, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
ROOT_DIR        = '/content/Radar-aided-beam-prediction-main/scenario9_dev'
CSV_FILE        = '/content/Radar-aided-beam-prediction-main/scenario9_dev/scenario9.csv'
NUM_CLIENTS     = 4
ROUNDS          = 30
LOCAL_EPOCHS    = 3
ATTN_EPOCHS     = 10      # Phase 2: attention fine-tuning epochs
BATCH_SIZE      = 32
LR              = 1e-3
ATTN_LR         = 5e-4
SEED            = 42
MAX_SAMPLES     = None    # None = all 5964
NUM_CLASSES     = 64
SAVE_DIR        = './saved_models/multimodal_FL/'

from dataset import LOADERS, load_mmwave_test
from models  import (ENCODERS, PredictionHead, ClientModel,
                     AttentionFusion, EMBED_DIM)

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device  : {device}')
print(f'Clients : Radar | Camera | LiDAR | GPS')
print(f'Fusion  : Attention (Phase 1: FL, Phase 2: Attention fine-tune)')
os.makedirs(SAVE_DIR, exist_ok=True)

MODALITY_NAMES = {0:'Radar', 1:'Camera', 2:'LiDAR', 3:'GPS-Cal'}

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────
print('\n=== Loading data ===')
client_data = {}

for cid in range(NUM_CLIENTS):
    print(f'\n[Client {cid} — {MODALITY_NAMES[cid]}]')
    X, y = LOADERS[cid](ROOT_DIR, CSV_FILE, max_samples=MAX_SAMPLES)
    print(f'  shape: {X.shape}  labels: {y.min()}-{y.max()}')
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)
    def tt(a, dt=torch.float32): return torch.tensor(a, dtype=dt)
    client_data[cid] = (
        tt(X_tr), tt(X_val), tt(X_te),
        tt(y_tr, torch.long), tt(y_val, torch.long), tt(y_te, torch.long),
    )
    print(f'  Train {len(X_tr)} | Val {len(X_val)} | Test {len(X_te)}')

# mmWave for comparison only
print('\n[mmWave — TEST ONLY]')
X_mm, y_mm = load_mmwave_test(ROOT_DIR, CSV_FILE, max_samples=MAX_SAMPLES)
_, X_tmp, _, y_tmp = train_test_split(X_mm, y_mm, test_size=0.3, random_state=SEED)
_, X_mm_te, _, y_mm_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)

# ─────────────────────────────────────────────────────────
# 2. BUILD MODELS
# ─────────────────────────────────────────────────────────
global_head    = PredictionHead(num_classes=NUM_CLASSES).to(device)
attention_net  = AttentionFusion(embed_dim=EMBED_DIM).to(device)
client_models  = {}

print('\n=== Building models ===')
for cid in range(NUM_CLIENTS):
    enc   = ENCODERS[cid]().to(device)
    model = ClientModel(enc, copy.deepcopy(global_head)).to(device)
    with torch.no_grad():
        model(client_data[cid][0][:2].to(device))
    client_models[cid] = model
    n = sum(p.numel() for p in model.parameters())
    print(f'  Client {cid} ({MODALITY_NAMES[cid]}) params: {n:,}')

n_attn = sum(p.numel() for p in attention_net.parameters())
print(f'  AttentionFusion params: {n_attn:,}')

# ─────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def iterate(X, y, bs=BATCH_SIZE, shuffle=True):
    idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for s in range(0, len(X), bs):
        e = min(s+bs, len(X))
        yield X[idx[s:e]], y[idx[s:e]]

def local_train(model, X_tr, y_tr):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    for _ in range(LOCAL_EPOCHS):
        for bx, by in iterate(X_tr, y_tr):
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            criterion(model(bx), by).backward()
            opt.step()

def fed_avg(states):
    avg = {}
    for k in states[0]:
        avg[k] = sum(s[k].float() for s in states) / len(states)
    return avg

def mmwave_baseline(X_mm, y_mm):
    X_t = torch.tensor(X_mm, dtype=torch.float32)
    y_t = torch.tensor(y_mm, dtype=torch.long)
    topk_idx = torch.topk(X_t, k=5, dim=1).indices
    hit = torch.zeros(len(y_t), dtype=torch.bool)
    accs = []
    for ki in range(5):
        hit |= (topk_idx[:, ki] == y_t)
        accs.append(hit.float().mean().item() * 100)
    return accs

# ─────────────────────────────────────────────────────────
# 4. COLLECT EMBEDDINGS (no raw data shared — only embeddings)
# ─────────────────────────────────────────────────────────
def collect_embeddings(split='val'):
    """
    Each client computes embeddings from their local data.
    Only embeddings (128-dim) are shared with server — NOT raw data.
    Privacy preserved: raw sensor data never leaves client.
    """
    si = 1 if split == 'val' else 2
    li = 4 if split == 'val' else 5
    n  = min(client_data[c][si].shape[0] for c in range(NUM_CLIENTS))
    y  = client_data[0][li][:n]

    all_embeddings = []  # list of (n, 128) per client
    for cid in range(NUM_CLIENTS):
        client_models[cid].eval()
        embs = []
        X = client_data[cid][si][:n]
        with torch.no_grad():
            for s in range(0, n, BATCH_SIZE):
                e = min(s+BATCH_SIZE, n)
                bx = X[s:e].to(device)
                emb = client_models[cid].encoder(bx)
                embs.append(emb.cpu())
        all_embeddings.append(torch.cat(embs))  # (n, 128)
    return all_embeddings, y[:n]

# ─────────────────────────────────────────────────────────
# 5. ATTENTION FUSION PREDICTION
# ─────────────────────────────────────────────────────────
def predict_with_attention(embeddings_per_client, available_clients=None):
    """
    embeddings_per_client: list of (B, 128) tensors, one per client
    available_clients: which clients are available (None = all)
    returns: logits (B, 64), attention_weights (B, N)
    """
    if available_clients is None:
        available_clients = list(range(NUM_CLIENTS))

    embs = [embeddings_per_client[c].to(device) for c in available_clients]

    attention_net.eval()
    global_head.eval()

    with torch.no_grad():
        fused, weights = attention_net(embs)    # (B, 128), (B, N)
        logits = global_head(fused)             # (B, 64)
    return logits, weights

def eval_attention_combo(available_clients, split='val'):
    """Evaluate using attention fusion for given subset of clients"""
    si = 1 if split == 'val' else 2
    li = 4 if split == 'val' else 5
    n  = min(client_data[c][si].shape[0] for c in available_clients)
    y  = client_data[available_clients[0]][li][:n]

    # Collect embeddings for available clients
    emb_dict = {}
    for cid in available_clients:
        client_models[cid].eval()
        embs = []
        X = client_data[cid][si][:n]
        with torch.no_grad():
            for s in range(0, n, BATCH_SIZE):
                e   = min(s+BATCH_SIZE, n)
                bx  = X[s:e].to(device)
                emb = client_models[cid].encoder(bx)
                embs.append(emb.cpu())
        emb_dict[cid] = torch.cat(embs)

    all_logits, all_labels = [], []
    for s in range(0, n, BATCH_SIZE):
        e    = min(s+BATCH_SIZE, n)
        embs = [emb_dict[c][s:e] for c in available_clients]
        attention_net.eval(); global_head.eval()
        with torch.no_grad():
            fused, _ = attention_net(embs)
            logits   = global_head(fused.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(y[s:e])

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    topk_idx   = torch.topk(all_logits, k=5, dim=1).indices
    hit = torch.zeros(len(all_labels), dtype=torch.bool)
    accs = []
    for k in range(5):
        hit |= (topk_idx[:, k] == all_labels)
        accs.append(hit.float().mean().item() * 100)
    return accs

# Simple eval without attention (for FL training phase)
def eval_simple(available_clients, split='val'):
    si = 1 if split == 'val' else 2
    li = 4 if split == 'val' else 5
    n  = min(client_data[c][si].shape[0] for c in available_clients)
    y  = client_data[available_clients[0]][li][:n]
    all_logits, all_labels = [], []
    for s in range(0, n, BATCH_SIZE):
        e     = min(s+BATCH_SIZE, n)
        embs  = []
        for c in available_clients:
            client_models[c].eval()
            with torch.no_grad():
                bx  = client_data[c][si][s:e].to(device)
                emb = client_models[c].encoder(bx)
            embs.append(emb)
        fused  = torch.stack(embs).mean(dim=0)
        global_head.eval()
        with torch.no_grad():
            logits = global_head(fused).cpu()
        all_logits.append(logits)
        all_labels.append(y[s:e])
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    topk_idx   = torch.topk(all_logits, k=5, dim=1).indices
    hit = torch.zeros(len(all_labels), dtype=torch.bool)
    accs = []
    for k in range(5):
        hit |= (topk_idx[:, k] == all_labels)
        accs.append(hit.float().mean().item() * 100)
    return accs

# ─────────────────────────────────────────────────────────
# 6. PHASE 1: FEDERATED TRAINING
# ─────────────────────────────────────────────────────────
mm_base = mmwave_baseline(X_mm_te, y_mm_te)
print(f'\nmmWave Baseline Top-5: {mm_base[4]:.2f}%  (upper bound)')

print('\n' + '='*55)
print('PHASE 1: Federated Training (30 rounds)')
print('='*55)

history   = {'round':[], 'top1':[], 'top5':[]}
best_top1 = 0.0
best_path = os.path.join(SAVE_DIR, 'best_shared_head.pth')

for r in range(ROUNDS):
    print(f'\n-- Round {r+1}/{ROUNDS} --')

    for cid in range(NUM_CLIENTS):
        client_models[cid].head.load_state_dict(
            copy.deepcopy(global_head.state_dict()))

    local_heads = []
    for cid in range(NUM_CLIENTS):
        X_tr,_,_, y_tr,_,_ = client_data[cid]
        local_train(client_models[cid], X_tr, y_tr)
        local_heads.append(copy.deepcopy(
            client_models[cid].head.state_dict()))

    global_head.load_state_dict(fed_avg(local_heads))
    for cid in range(NUM_CLIENTS):
        client_models[cid].head.load_state_dict(
            global_head.state_dict())

    accs = eval_simple(list(range(NUM_CLIENTS)), 'val')
    print(f'  Val  Top-1: {accs[0]:.2f}%  Top-5: {accs[4]:.2f}%')

    history['round'].append(r+1)
    history['top1'].append(accs[0])
    history['top5'].append(accs[4])

    if accs[0] > best_top1:
        best_top1 = accs[0]
        torch.save(global_head.state_dict(), best_path)
        for cid in range(NUM_CLIENTS):
            torch.save(client_models[cid].encoder.state_dict(),
                       os.path.join(SAVE_DIR,
                       f'best_encoder_client{cid}.pth'))
        print(f'  ✓ Best saved ({best_top1:.2f}%)')

print(f'\nPhase 1 Best Top-1: {best_top1:.2f}%')

# Load best FL model
global_head.load_state_dict(torch.load(best_path))
for cid in range(NUM_CLIENTS):
    client_models[cid].encoder.load_state_dict(
        torch.load(os.path.join(SAVE_DIR,
        f'best_encoder_client{cid}.pth')))
    client_models[cid].head.load_state_dict(
        global_head.state_dict())

# ─────────────────────────────────────────────────────────
# 7. PHASE 2: ATTENTION FINE-TUNING
# ─────────────────────────────────────────────────────────
print('\n' + '='*55)
print('PHASE 2: Attention Fine-Tuning')
print('Server collects embeddings (NOT raw data) from all clients')
print('Trains AttentionFusion network on those embeddings')
print('='*55)

# Collect embeddings from ALL clients on validation set
# Note: only 128-dim embeddings shared, not raw data
print('\nCollecting embeddings from clients (privacy-safe)...')
val_embeddings, val_labels = collect_embeddings('val')
print(f'  Embeddings shape: {val_embeddings[0].shape} per client')
print(f'  Labels shape    : {val_labels.shape}')

# Freeze encoders and head — only train attention network
for cid in range(NUM_CLIENTS):
    for p in client_models[cid].encoder.parameters():
        p.requires_grad = False
for p in global_head.parameters():
    p.requires_grad = False

attn_optimizer = torch.optim.Adam(
    attention_net.parameters(), lr=ATTN_LR, weight_decay=1e-4)

best_attn_top1 = 0.0
attn_path      = os.path.join(SAVE_DIR, 'best_attention.pth')
attn_history   = {'epoch':[], 'top1':[], 'top5':[]}

print(f'\nFine-tuning AttentionFusion for {ATTN_EPOCHS} epochs...')
n_val = val_embeddings[0].shape[0]

for epoch in range(ATTN_EPOCHS):
    attention_net.train()
    global_head.eval()

    # Shuffle indices
    idx = torch.randperm(n_val)
    total_loss = 0.0

    for s in range(0, n_val, BATCH_SIZE):
        e     = min(s+BATCH_SIZE, n_val)
        batch = idx[s:e]

        embs  = [val_embeddings[c][batch].to(device)
                 for c in range(NUM_CLIENTS)]
        by    = val_labels[batch].to(device)

        attn_optimizer.zero_grad()

        # Forward: attention fusion → head → loss
        fused, weights = attention_net(embs)   # (B,128), (B,4)
        logits         = global_head(fused)    # (B,64)
        loss           = criterion(logits, by)

        loss.backward()
        attn_optimizer.step()
        total_loss += loss.item()

    # Evaluate
    accs = eval_attention_combo(list(range(NUM_CLIENTS)), 'val')
    avg_loss = total_loss / (n_val // BATCH_SIZE + 1)
    print(f'  Epoch {epoch+1:2d}/{ATTN_EPOCHS}  '
          f'Loss: {avg_loss:.4f}  '
          f'Top-1: {accs[0]:.2f}%  '
          f'Top-5: {accs[4]:.2f}%')

    attn_history['epoch'].append(epoch+1)
    attn_history['top1'].append(accs[0])
    attn_history['top5'].append(accs[4])

    if accs[0] > best_attn_top1:
        best_attn_top1 = accs[0]
        torch.save(attention_net.state_dict(), attn_path)
        print(f'  ✓ Best attention saved ({best_attn_top1:.2f}%)')

print(f'\nPhase 2 Best Attention Top-1: {best_attn_top1:.2f}%')

# Unfreeze for potential future use
for cid in range(NUM_CLIENTS):
    for p in client_models[cid].encoder.parameters():
        p.requires_grad = True
for p in global_head.parameters():
    p.requires_grad = True

# Load best attention
attention_net.load_state_dict(torch.load(attn_path))

# ─────────────────────────────────────────────────────────
# 8. FINAL TEST — attention vs simple vs individual
# ─────────────────────────────────────────────────────────
print('\n' + '='*55)
print('FINAL TEST RESULTS')
print('='*55)
print('\nExpected order: ALL(Attn) > subsets > singles > LiDAR\n')

combos = {
    'ALL 4 — Attention Fusion' : [0,1,2,3],
    'Radar+Camera — Attention' : [0,1],
    'Camera+GPS — Attention'   : [1,3],
    'Camera only'              : [1],
    'Radar only'               : [0],
    'GPS only'                 : [3],
    'LiDAR only'               : [2],
}

print(f'{"Combination":<32} {"Top-1":>8} {"Top-2":>8} '
      f'{"Top-3":>8} {"Top-5":>8}')
print('-' * 65)

all_results = {}
for name, clients in combos.items():
    accs = eval_attention_combo(clients, 'test')
    all_results[name] = accs
    print(f'{name:<32} {accs[0]:>7.2f}% {accs[1]:>7.2f}% '
          f'{accs[2]:>7.2f}% {accs[4]:>7.2f}%')

print(f'\n{"mmWave Baseline (NOT input)":<32} '
      f'{mm_base[0]:>7.2f}% {mm_base[1]:>7.2f}% '
      f'{mm_base[2]:>7.2f}% {mm_base[4]:>7.2f}%')

# Show average attention weights per modality (interpretability)
print('\n=== Attention Weight Analysis ===')
print('(How much each modality contributes on average)\n')
all_embeddings_test, test_labels = collect_embeddings('test')
_, w_sample = predict_with_attention(all_embeddings_test)
avg_weights = w_sample.mean(dim=0).cpu().numpy()
for cid in range(NUM_CLIENTS):
    print(f'  {MODALITY_NAMES[cid]:10s}: avg attention = {avg_weights[cid]:.4f}  '
          f'({avg_weights[cid]*100:.1f}%)')

# ─────────────────────────────────────────────────────────
# 9. SAVE + PLOT
# ─────────────────────────────────────────────────────────
full_history = {'phase1': history, 'phase2': attn_history}
with open(os.path.join(SAVE_DIR, 'history.json'), 'w') as f:
    json.dump(full_history, f, indent=2)

try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Phase 1 FL convergence
    ax = axes[0]
    ax.plot(history['round'], history['top1'], 'b-o',
            label='Top-1 (equal fusion)')
    ax.plot(history['round'], history['top5'], 'g-o',
            label='Top-5 (equal fusion)')
    ax.axhline(mm_base[4], color='orange', linestyle='--',
               label=f'mmWave Top-5 ({mm_base[4]:.1f}%)')
    ax.set_xlabel('FL Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Phase 1: FL Training Convergence')
    ax.legend(fontsize=8); ax.grid(True)

    # Plot 2: Phase 2 attention fine-tuning
    ax = axes[1]
    ax.plot(attn_history['epoch'], attn_history['top1'],
            'r-o', label='Top-1 (attention fusion)')
    ax.plot(attn_history['epoch'], attn_history['top5'],
            'm-o', label='Top-5 (attention fusion)')
    ax.axhline(best_top1, color='blue', linestyle='--',
               label=f'Phase 1 best ({best_top1:.1f}%)')
    ax.set_xlabel('Attention Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Phase 2: Attention Fine-Tuning')
    ax.legend(fontsize=8); ax.grid(True)

    # Plot 3: Modality attention weights
    ax = axes[2]
    names  = [MODALITY_NAMES[i] for i in range(NUM_CLIENTS)]
    colors = ['steelblue', 'darkorange', 'green', 'red']
    bars   = ax.bar(names, avg_weights * 100, color=colors, alpha=0.8)
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set_ylabel('Average Attention Weight (%)')
    ax.set_title('Learned Attention Weights per Modality')
    ax.set_ylim(0, max(avg_weights)*120)
    ax.grid(True, axis='y')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'attention_fusion_results.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f'\nPlot saved -> {path}')
except Exception as e:
    print(f'Plot skipped: {e}')

print('\nDone!')