#!/usr/bin/env python3
"""
HessGPT Pre-Training — LLaMA-3 Tokenizer

USAGE:
    python pretrain.py

LOGIQUE EPOCHS & CHUNKS :
    - 1 chunk chargé en RAM à la fois (~10GB), trainé, libéré
    - Epoch 1 → chunk_000, chunk_001, chunk_002, chunk_003
    - Epoch 2 → chunk_004, chunk_005, chunk_006, chunk_007
    - Sliding window strict : jamais de recyclage de chunk
    - S'arrête automatiquement si plus assez de chunks dispo
    - num_epochs dans CONFIG = maximum souhaité

SAVES :
    - 25%         : après chaque chunk complet
    - Intra-chunk : tous les save_every_steps optimizer steps (safety net)
    - Fin epoch   : automatique
    - CTRL+C / exception : immédiat

REPRISE :
    Relancer la même commande — le .pt est chargé automatiquement.

CONTENU DU .pt :
    model_state_dict      poids float32
    optimizer_state_dict  moments AdamW
    scheduler_state_dict  current_step WSD
    current_epoch         epoch en cours (base-1)
    chunk_within_epoch    prochain cwi à traiter (base-0)
    global_step           total optimizer steps
    total_training_time   secondes GPU accumulées
    config                CONFIG complet
    training_history      dict complet
    last_save             timestamp ISO

BUGS FIXÉS v2 :
    - configure_optimizers : itération directe sur named_parameters() du modèle
      non-compilé pour éviter le KeyError causé par la double-itération
      named_modules() + named_parameters(recurse=True).
    - LazyChunkDataset : val_tokens correctement stocké comme self.val_tokens
      et utilisé dans _load() via self.val_tokens (NameError corrigé).
"""

import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import math
import json
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # pas de display — sauvegarde fichier direct (compatible SSH/tmux)
import matplotlib.pyplot as plt

from huggingface_hub import login
login(token="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")

_core = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Core')
sys.path.append(os.path.join(_core, 'Model'))
sys.path.append(os.path.join(_core, 'Attention'))
sys.path.append(os.path.join(_core, 'FeedForward'))
sys.path.append(os.path.join(_core, 'TransformerBlock'))

# ============================================================
# TOKENS SPÉCIAUX
# ============================================================
SPECIAL_TOKENS = ['<code>', '<think>', '</think>']

print("=" * 80)
print("HessGPT v5 — LLaMA-3 | RMSNorm | Flash | QK-Norm | WSD | Soft-cap")
print("=" * 80)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # ── Modèle
    'vocab_size':            None,
    'embed_dim':             1280,
    'num_heads':             20,
    'num_layers':            24,
    'max_seq_len':           1024,
    'dropout':               0.0,
    'use_rope':              True,
    'use_yarn':              False,
    'yarn_scale':            4.0,
    'yarn_original_max_len': 1024,
    'use_swiglu':            True,
    'n_kv_heads':            5,
    'use_qk_norm':           True,
    'soft_cap':              30.0,
    'use_flash_attn':        True,

    # ── Training
    'batch_size':            32,
    'gradient_accumulation': 4,
    'max_grad_norm':         1.0,
    'learning_rate':         4e-4,
    'weight_decay':          0.1,
    'adam_beta1':            0.9,
    'adam_beta2':            0.95,
    'adam_eps':              1e-8,

    # ── Epochs & Chunks
    'num_epochs':       5,
    'chunks_per_epoch': 3,

    # ── Data
    'data_dir':   './data/ultra_filtered',
    'val_tokens': 15_000_000,

    # ── WSD LR Schedule
    'warmup_ratio': 0.03,
    'decay_ratio':  0.15,
    'min_lr_ratio': 0.1,

    # ── Validation
    'validate_every_steps': 500,
    'val_batches':          50,

    # ── Saves
    'save_every_steps': 2000,

    # ── Checkpoint
    'checkpoint_file': './checkpoints/HessGpt_pretrain.pt',

    # ── System
    'use_compile':  True,
    'compile_mode': 'default',
    'num_workers':  4,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nCONFIG :")
print(f"  embed={CONFIG['embed_dim']}  layers={CONFIG['num_layers']}  "
      f"heads={CONFIG['num_heads']}  kv={CONFIG['n_kv_heads']}")
print(f"  epochs={CONFIG['num_epochs']}  chunks/epoch={CONFIG['chunks_per_epoch']}")
print(f"  batch={CONFIG['batch_size']}  accum={CONFIG['gradient_accumulation']}  "
      f"device={device}")
if device == 'cuda':
    print(f"  GPU={torch.cuda.get_device_name(0)}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

# ============================================================
# SCAN CHUNKS
# ============================================================
def scan_available_chunks(data_dir):
    available = []
    if not os.path.exists(data_dir):
        return available
    for entry in sorted(os.listdir(data_dir)):
        if not entry.startswith('chunk'):
            continue
        chunk_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(chunk_dir):
            continue
        stats_file = os.path.join(chunk_dir, 'stats.json')
        if not os.path.exists(stats_file):
            continue
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            npy_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.npy')])
            if not npy_files:
                continue
            cid = int(entry.split('_')[1]) if '_' in entry else int(entry.replace('chunk', ''))
            available.append({
                'id':    cid,
                'dir':   chunk_dir,
                'files': npy_files,
                'stats': stats,
            })
        except Exception as e:
            print(f"  skip {entry}: {e}")
    available.sort(key=lambda x: x['id'])
    return available

print(f"\nScan chunks...")
ALL_CHUNKS = scan_available_chunks(CONFIG['data_dir'])
n_chunks   = len(ALL_CHUNKS)
print(f"  {n_chunks} chunks trouvés")

if n_chunks == 0:
    print(f"ERREUR: aucun chunk dans {CONFIG['data_dir']}")
    sys.exit(1)

epochs_possible = n_chunks // CONFIG['chunks_per_epoch']
if epochs_possible == 0:
    print(f"  ERREUR: {n_chunks} chunks < chunks_per_epoch={CONFIG['chunks_per_epoch']}")
    sys.exit(1)

if epochs_possible < CONFIG['num_epochs']:
    print(f"  WARN: seulement {epochs_possible} epochs possibles avec {n_chunks} chunks")
    CONFIG['num_epochs'] = epochs_possible

TOTAL_CHUNKS_USED = CONFIG['num_epochs'] * CONFIG['chunks_per_epoch']
ALL_TRAIN_CHUNKS  = ALL_CHUNKS[:TOTAL_CHUNKS_USED]

print(f"\nPLAN CHUNKS (sliding window, jamais recyclés) :")
for ep in range(CONFIG['num_epochs']):
    s   = ep * CONFIG['chunks_per_epoch']
    ids = [f"chunk_{c['id']:03d}" for c in ALL_TRAIN_CHUNKS[s:s + CONFIG['chunks_per_epoch']]]
    print(f"  Epoch {ep+1:2d} : {' '.join(ids)}")

# ============================================================
# CALCUL STEPS
# ============================================================
def steps_for_chunk(stats):
    samples = stats['total_tokens'] // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    return max(math.ceil(batches / CONFIG['gradient_accumulation']), 1)

TOTAL_STEPS     = sum(steps_for_chunk(c['stats']) for c in ALL_TRAIN_CHUNKS)
STEPS_PER_EPOCH = TOTAL_STEPS // CONFIG['num_epochs']

print(f"\nPLAN STEPS :")
print(f"  steps/epoch~={STEPS_PER_EPOCH:,}  total={TOTAL_STEPS:,}")
print(f"  save 25% : après chaque chunk")
print(f"  safety save : tous les {CONFIG['save_every_steps']} steps")

# ============================================================
# TOKENIZER
# ============================================================
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3-8B")
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
print(f"  vocab={len(tokenizer)}")
for tok in SPECIAL_TOKENS:
    print(f"  {tok} → {tokenizer.convert_tokens_to_ids(tok)}")

# ============================================================
# WSD SCHEDULER
# ============================================================
class WSDScheduler:
    """
    WSD schedule partagé entre Muon et AdamW.

    Muon utilise un lr fixe (lr_muon = lr_adamw * 5) — le ratio est
    conservé à chaque step : quand AdamW est à X%, Muon est aussi à X%
    de son lr max.

    optimizers : liste [muon_opt, adamw_opt] ou un seul optimizer
    """
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.03, decay_ratio=0.15, min_lr_ratio=0.1):
        # Accepte un seul optimizer ou une liste
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0
        print(f"\nWSD LR : warmup={self.warmup_steps:,}  "
              f"stable={self.stable_steps:,}  decay={self.decay_steps:,}")
        print(f"  AdamW {self.min_lr:.2e} → {self.max_lr:.2e}  "
              f"| Muon {self.min_lr*5:.2e} → {self.max_lr*5:.2e}")

    def get_lr(self):
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self):
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                # Muon garde son ratio 5x — on détecte via la présence de 'momentum'
                if 'momentum' in pg:
                    pg['lr'] = lr * 5.0
                else:
                    pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd['current_step']

# ============================================================
# GRAPHES — Loss & PPL curves
# ============================================================
def plot_loss_curve(training_history, save_path=None):
    """
    Graph loss train (raw + smooth 200 steps) + val loss par optimizer step.
    Sauvegardé en PNG, mis à jour à chaque validation et en fin de training.
    """
    steps      = [e['step']       for e in training_history.get('loss_curve', [])]
    train_loss = [e['train_loss'] for e in training_history.get('loss_curve', [])]
    val_steps  = [e['step']       for e in training_history.get('validations', [])]
    val_loss   = [e['val_loss']   for e in training_history.get('validations', [])]

    if not steps:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    # Courbe brute
    ax.plot(steps, train_loss, color='steelblue', linewidth=0.8,
            alpha=0.4, label='Train loss (raw)')

    # Moyenne glissante 200 steps
    # np.convolve mode='valid' → len = N - K + 1, aligné sur steps[K-1:]
    if len(train_loss) >= 200:
        K        = 200
        kernel   = np.ones(K) / K
        smoothed = np.convolve(train_loss, kernel, mode='valid')  # len = N - K + 1
        ax.plot(steps[K - 1:], smoothed,
                color='steelblue', linewidth=2.0, label='Train loss (smooth ×200)')

    # Val loss
    if val_steps:
        ax.plot(val_steps, val_loss, color='tomato', linewidth=2.0,
                marker='o', markersize=4, label='Val loss')

    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Loss')
    ax.set_title('HessGPT Pretrain — Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = CONFIG['checkpoint_file'].replace('.pt', '_loss.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📈 Loss curve → {save_path}")


def plot_ppl_curve(training_history, save_path=None):
    """
    Graph perplexité train (smooth 200 steps) + val ppl par optimizer step.
    PPL = exp(loss), clippée à 1000 pour éviter les pics NaN en début de training.
    """
    steps      = [e['step']       for e in training_history.get('loss_curve', [])]
    train_loss = [e['train_loss'] for e in training_history.get('loss_curve', [])]
    val_steps  = [e['step']       for e in training_history.get('validations', [])]
    val_ppl    = [e['val_ppl']    for e in training_history.get('validations', [])]

    if not steps:
        return

    # PPL train — seulement la version smooth pour lisibilité
    train_ppl = [math.exp(min(l, 6.9)) for l in train_loss]  # clip à ~1000 PPL max

    fig, ax = plt.subplots(figsize=(14, 5))

    if len(train_ppl) >= 200:
        K        = 200
        kernel   = np.ones(K) / K
        smoothed = np.convolve(train_ppl, kernel, mode='valid')  # len = N - K + 1
        ax.plot(steps[K - 1:], smoothed,
                color='mediumseagreen', linewidth=2.0, label='Train PPL (smooth ×200)')
    else:
        ax.plot(steps, train_ppl, color='mediumseagreen', linewidth=1.5,
                label='Train PPL (raw)')

    if val_steps:
        ax.plot(val_steps, val_ppl, color='darkorange', linewidth=2.0,
                marker='o', markersize=4, label='Val PPL')

    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Perplexity')
    ax.set_title('HessGPT Pretrain — Perplexity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = CONFIG['checkpoint_file'].replace('.pt', '_ppl.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 PPL curve  → {save_path}")


def save_graphs(training_history):
    """Sauvegarde les deux graphes en une seule appel."""
    plot_loss_curve(training_history)
    plot_ppl_curve(training_history)


# ============================================================
# DATASET — 1 chunk à la fois en RAM
# ============================================================
class ChunkSubset(Dataset):
    """Séquences extraites d'un tensor de tokens en RAM."""
    def __init__(self, tokens, seq_len, pad_token_id):
        self.tokens       = tokens
        self.seq_len      = seq_len
        self.pad_token_id = pad_token_id
        self.num_samples  = len(tokens) // (seq_len + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad   = torch.full((self.seq_len + 1 - len(chunk),),
                               self.pad_token_id, dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk[:-1], chunk[1:]


class LazyChunkDataset:
    """
    Charge 1 chunk en RAM, split train/val.
    Appeler unload() après le train pour libérer la RAM.

    ✅ FIX BUG 5 : val_tokens est maintenant stocké dans self.val_tokens
    et utilisé via self.val_tokens dans _load(). L'ancienne version utilisait
    val_tokens directement dans _load() sans le passer en argument ni le
    stocker, causant un NameError garanti à l'exécution.
    """
    def __init__(self, chunk_info, seq_len, pad_token_id, val_tokens=15_000_000):
        self.seq_len      = seq_len
        self.pad_token_id = pad_token_id
        self.val_tokens   = val_tokens   # ✅ FIX : stocké comme attribut
        self._load(chunk_info)

    def _load(self, chunk_info):
        print(f"  Loading chunk_{chunk_info['id']:03d}...")
        t0 = time.time()

        arrays = []
        for fname in chunk_info['files']:
            fpath = os.path.join(chunk_info['dir'], fname)
            try:
                arrays.append(
                    torch.from_numpy(np.load(fpath).copy()).long()
                )
            except Exception as e:
                print(f"    skip {fname}: {e}")

        if not arrays:
            raise ValueError(f"chunk_{chunk_info['id']:03d} : aucun fichier chargé")

        all_tokens = torch.cat(arrays)
        total      = len(all_tokens)
        val_size   = min(self.val_tokens, int(total * 0.05))  # ✅ FIX : self.val_tokens
        train_size = total - val_size

        self._train_toks = all_tokens[:train_size]
        self._val_toks   = all_tokens[train_size:]

        ram_gb = all_tokens.element_size() * total / 1e9
        print(f"  chunk_{chunk_info['id']:03d} : {total/1e6:.0f}M tokens  "
              f"train={train_size/1e6:.0f}M  val={val_size/1e6:.0f}M  "
              f"RAM={ram_gb:.1f}GB  ({time.time()-t0:.1f}s)")

    def get_train_dataset(self):
        return ChunkSubset(self._train_toks, self.seq_len, self.pad_token_id)

    def get_val_dataset(self):
        return ChunkSubset(self._val_toks, self.seq_len, self.pad_token_id)

    def unload(self):
        del self._train_toks, self._val_toks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  RAM chunk libérée")

# ============================================================
# CHECKPOINT MANAGER
# ============================================================
class CheckpointManager:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata):
        """Sauvegarde atomique : écrit dans .tmp puis os.replace.
        optimizers : tuple (muon_opt, adamw_opt) ou optimizer unique.
        """
        m  = model._orig_mod if hasattr(model, '_orig_mod') else model
        if isinstance(optimizers, (list, tuple)):
            muon_opt, adamw_opt = optimizers
            opt_state = {
                'muon_state_dict':  muon_opt.state_dict(),
                'adamw_state_dict': adamw_opt.state_dict(),
            }
        else:
            opt_state = {'optimizer_state_dict': optimizers.state_dict()}
        cp = {
            **opt_state,
            'model_state_dict':     m.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_epoch':        metadata['current_epoch'],
            'chunk_within_epoch':   metadata['chunk_within_epoch'],
            'global_step':          metadata['global_step'],
            'total_training_time':  metadata.get('total_training_time', 0.0),
            'config':               CONFIG,
            'training_history':     metadata['training_history'],
            'last_save':            datetime.now().isoformat(),
        }
        tmp = self.path + '.tmp'
        torch.save(cp, tmp)
        os.replace(tmp, self.path)
        print(f"  💾 SAVE → epoch={metadata['current_epoch']}  "
              f"next_cwi={metadata['chunk_within_epoch']}/{CONFIG['chunks_per_epoch']}  "
              f"step={metadata['global_step']:,}  [{self.path}]")

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\nCheckpoint trouvé : {self.path}")
        cp = torch.load(self.path, map_location='cpu', weights_only=False)
        print(f"  epoch={cp.get('current_epoch','?')}  "
              f"cwi={cp.get('chunk_within_epoch','?')}  "
              f"step={cp.get('global_step',0):,}  "
              f"saved={cp.get('last_save','?')}")
        return cp

# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate(model, val_loader, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
            total_loss += loss.item()
            n += 1
    finally:
        model.train()
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 10)), avg

# ============================================================
# MUON OPTIMIZER
# ============================================================
# Muon (MomentUm Orthogonalized by Newton-Schulz) applique une
# mise à jour orthogonalisée sur les matrices 2D des couches cachées.
# Règle d'or :
#   - Muon  → matrices 2D des projections attention + FFN
#   - AdamW → embeddings, output head, RMSNorm, biases, inv_freq RoPE
#
# Référence : https://github.com/KellerJordan/modded-nanogpt
# ============================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximation Newton-Schulz de la matrice orthogonale la plus proche de G.
    Converge en ~5 itérations pour des matrices bien conditionnées.
    Retourne X tel que X^T X ≈ I et X G^T ≈ polar(G).
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Normalisation initiale pour la stabilité numérique
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-Schulz.

    Appliqué UNIQUEMENT sur les matrices 2D des couches cachées
    (projections attention Q/K/V/O et FFN gate/up/down).

    Ne jamais utiliser sur :
      - token_embeddings  (gradients clairsemés)
      - output_head       (weight-tied avec embeddings)
      - RMSNorm weights   (tenseurs 1D)
      - inv_freq RoPE     (buffer non-entraînable)

    Args:
        params        : paramètres 2D des couches cachées
        lr            : learning rate (défaut 0.02, ~5x AdamW)
        momentum      : momentum Nesterov (défaut 0.95)
        nesterov      : utiliser Nesterov momentum (recommandé)
        ns_steps      : itérations Newton-Schulz (5 suffit)
        weight_decay  : weight decay (défaut 0.0 — Muon orthogonalise déjà)
    """
    def __init__(self, params, lr=0.02, momentum=0.95,
                 nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr           = group['lr']
            momentum     = group['momentum']
            nesterov     = group['nesterov']
            ns_steps     = group['ns_steps']
            wd           = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if g.ndim < 2:
                    # Sécurité : ne jamais orthogonaliser un vecteur 1D
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g + momentum * buf
                else:
                    g = buf

                # Orthogonalisation via Newton-Schulz
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Mise à l'échelle : norme de Frobenius normalisée
                scale = max(g.size(0), g.size(1)) ** 0.5
                g = g * scale

                # Weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(g, alpha=-lr)


# ============================================================
# CONFIGURE OPTIMIZERS — Muon + AdamW
# ============================================================
def configure_optimizers(model, lr, weight_decay, betas, eps):
    """
    Partitionnement des paramètres :

      muon_params  : matrices 2D des couches cachées (attention + FFN)
                     → Muon avec lr_muon ≈ 5x lr AdamW
      adamw_decay  : autres matrices 2D (embeddings, output_head)
                     → AdamW avec weight decay
      adamw_nodecay: tenseurs 1D (RMSNorm, biases, inv_freq)
                     → AdamW sans weight decay

    Règle d'exclusion Muon :
      - token_embeddings.weight  → gradients clairsemés (sparse)
      - output_head.weight       → weight-tied avec embeddings
      - Tout paramètre hors des TransformerBlocks

    Args:
        model : modèle NON compilé (unwrap _orig_mod avant d'appeler)
    """
    # Noms des paramètres à exclure de Muon
    MUON_EXCLUDE = {
        'token_embeddings.weight',
        'output_head.weight',
        'position_embeddings.weight',
    }

    muon_params   = []   # matrices 2D dans les blocks → Muon
    adamw_decay   = []   # matrices 2D hors blocks → AdamW + wd
    adamw_nodecay = []   # tenseurs 1D → AdamW sans wd

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Paramètres explicitement exclus de Muon
        if pn in MUON_EXCLUDE:
            if p.dim() >= 2:
                adamw_decay.append(p)
            else:
                adamw_nodecay.append(p)
            continue

        if p.dim() >= 2 and pn.startswith('blocks.'):
            # Matrice 2D dans un TransformerBlock → Muon
            muon_params.append(p)
        elif p.dim() >= 2:
            # Matrice 2D hors block (ln_final.weight si 2D, etc.)
            adamw_decay.append(p)
        else:
            # Tenseur 1D (norms, biases, inv_freq)
            adamw_nodecay.append(p)

    # lr Muon ≈ 5x lr AdamW (standard recommandé)
    lr_muon = lr * 5.0

    muon_opt = Muon(
        muon_params,
        lr           = lr_muon,
        momentum     = 0.95,
        nesterov     = True,
        ns_steps     = 5,
        weight_decay = 0.0,   # Muon orthogonalise déjà — wd inutile
    )

    adamw_opt = torch.optim.AdamW(
        [
            {'params': adamw_decay,   'weight_decay': weight_decay},
            {'params': adamw_nodecay, 'weight_decay': 0.0},
        ],
        lr    = lr,
        betas = betas,
        eps   = eps,
        fused = (device == 'cuda'),
    )

    print(f"\nOptimizer Muon+AdamW :")
    print(f"  Muon   : {len(muon_params):3d} tenseurs  lr={lr_muon:.2e}")
    print(f"  AdamW  : {len(adamw_decay):3d} tenseurs decay + "
          f"{len(adamw_nodecay):3d} no-decay  lr={lr:.2e}")

    return muon_opt, adamw_opt

# ============================================================
# TRAIN ONE CHUNK
# ============================================================
def train_one_chunk(
    model, chunk_info,
    optimizers, scheduler,
    checkpoint_manager, training_history,
    global_step, total_training_time,
    current_epoch, chunk_within_epoch,
):
    """
    optimizers : tuple (muon_opt, adamw_opt)
    """
    muon_opt, adamw_opt = optimizers

    label = (f"Epoch {current_epoch}/{CONFIG['num_epochs']} | "
             f"cwi {chunk_within_epoch+1}/{CONFIG['chunks_per_epoch']} "
             f"(chunk_{chunk_info['id']:03d})")

    print(f"\n{'='*80}")
    print(f"  {label}  LR_adamw={scheduler.get_last_lr()[0]:.2e}"
          f"  LR_muon={scheduler.get_last_lr()[0]*5:.2e}")
    print(f"{'='*80}")

    try:
        cds = LazyChunkDataset(
            chunk_info, CONFIG['max_seq_len'],
            tokenizer.pad_token_id, CONFIG['val_tokens'],
        )
    except Exception as e:
        print(f"  ERREUR chargement chunk : {e}")
        return global_step, total_training_time

    train_loader = DataLoader(
        cds.get_train_dataset(),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        cds.get_val_dataset(),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    num_batches = len(train_loader)
    print(f"  train={num_batches:,} batches | val={len(val_loader):,} batches")

    model.train()
    chunk_loss        = 0.0
    valid_batches     = 0
    accumulated_steps = 0
    running_loss      = 0.0
    running_batches   = 0
    t_start           = time.time()
    ae  = (device == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32

    pbar = tqdm(train_loader, desc=label, leave=True)

    for batch_idx, (x, y) in enumerate(pbar):
        try:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
                loss = loss / CONFIG['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                continue

            loss.backward()
            accumulated_steps += 1

            is_last = (batch_idx + 1 == num_batches)
            if (accumulated_steps % CONFIG['gradient_accumulation'] == 0) or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                muon_opt.step()
                adamw_opt.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated_steps = 0
                global_step += 1

                if global_step % CONFIG['validate_every_steps'] == 0:
                    val_ppl, val_loss = validate(model, val_loader, CONFIG['val_batches'])
                    avg = running_loss / max(running_batches, 1)
                    print(f"\n  step={global_step:,} | "
                          f"train={avg:.4f} ppl={math.exp(min(avg,10)):.1f} | "
                          f"val={val_loss:.4f} ppl={val_ppl:.1f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e}\n")
                    training_history['validations'].append({
                        'step':               global_step,
                        'current_epoch':      current_epoch,
                        'chunk_within_epoch': chunk_within_epoch,
                        'chunk_id':           chunk_info['id'],
                        'val_loss':           val_loss,
                        'val_ppl':            val_ppl,
                        'train_loss':         avg,
                        'lr':                 scheduler.get_last_lr()[0],
                    })
                    save_graphs(training_history)

                if global_step % CONFIG['save_every_steps'] == 0:
                    checkpoint_manager.save(model, optimizers, scheduler, metadata={
                        'current_epoch':       current_epoch,
                        'chunk_within_epoch':  chunk_within_epoch,
                        'global_step':         global_step,
                        'total_training_time': total_training_time + (time.time() - t_start),
                        'training_history':    training_history,
                    })

            raw = loss.item() * CONFIG['gradient_accumulation']
            chunk_loss      += raw
            running_loss    += raw
            valid_batches   += 1
            running_batches += 1

            # ── Loss curve logging (1 point par optimizer step) ──
            # On log uniquement quand un optimizer step vient d'avoir lieu
            # accumulated_steps == 0 signifie qu'on vient de faire step()
            if accumulated_steps == 0 and global_step > 0:
                training_history.setdefault('loss_curve', []).append({
                    'step':       global_step,
                    'train_loss': raw,
                })

            if batch_idx % 20 == 0:
                avg = running_loss / max(running_batches, 1)
                pbar.set_postfix(
                    loss=f'{raw:.4f}',
                    avg=f'{avg:.4f}',
                    ppl=f'{math.exp(min(avg,10)):.1f}',
                    lr=f'{scheduler.get_last_lr()[0]:.2e}',
                    step=f'{global_step:,}',
                )

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  OOM batch {batch_idx} — skip")
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated_steps = 0
                gc.collect()
                model.train()
                continue
            raise

    pbar.close()
    elapsed = time.time() - t_start
    total_training_time += elapsed
    avg_loss = chunk_loss / max(valid_batches, 1)
    print(f"\n  chunk_{chunk_info['id']:03d} terminé | "
          f"loss={avg_loss:.4f} | {elapsed/60:.1f}min")

    training_history['chunks'].append({
        'current_epoch':      current_epoch,
        'chunk_within_epoch': chunk_within_epoch,
        'chunk_id':           chunk_info['id'],
        'train_loss':         avg_loss,
        'time_sec':           elapsed,
        'batches':            valid_batches,
        'global_step':        global_step,
    })

    cds.unload()
    del cds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step, total_training_time

# ============================================================
# MAIN
# ============================================================
def main():
    from HessGpt import HessGPT

    print('\n' + '='*80 + '\nCREATION MODELE\n' + '='*80)

    ckpt_mgr = CheckpointManager(CONFIG['checkpoint_file'])

    model = HessGPT(
        vocab_size            = CONFIG['vocab_size'],
        embed_dim             = CONFIG['embed_dim'],
        num_heads             = CONFIG['num_heads'],
        num_layers            = CONFIG['num_layers'],
        max_seq_len           = CONFIG['max_seq_len'],
        dropout               = CONFIG['dropout'],
        use_rope              = CONFIG['use_rope'],
        use_yarn              = CONFIG['use_yarn'],
        yarn_scale            = CONFIG['yarn_scale'],
        yarn_original_max_len = CONFIG['yarn_original_max_len'],
        use_swiglu            = CONFIG['use_swiglu'],
        n_kv_heads            = CONFIG['n_kv_heads'],
        use_qk_norm           = CONFIG['use_qk_norm'],
        soft_cap              = CONFIG['soft_cap'],
        use_flash_attn        = CONFIG['use_flash_attn'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total_params/1e6:.1f}M")

    if CONFIG['use_compile'] and device == 'cuda':
        print('torch.compile...')
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL (on continue sans) : {e}')

    # Unwrap modèle compilé avant configure_optimizers
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizers = configure_optimizers(
        raw_model,
        CONFIG['learning_rate'],
        CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']),
        CONFIG['adam_eps'],
    )
    muon_opt, adamw_opt = optimizers

    scheduler = WSDScheduler(
        list(optimizers),
        max_lr        = CONFIG['learning_rate'],
        total_steps   = TOTAL_STEPS,
        warmup_ratio  = CONFIG['warmup_ratio'],
        decay_ratio   = CONFIG['decay_ratio'],
        min_lr_ratio  = CONFIG['min_lr_ratio'],
    )

    training_history = {
        'config':       CONFIG,
        'total_params': total_params,
        'total_steps':  TOTAL_STEPS,
        'chunks':       [],
        'validations':  [],
        'epochs':       [],
        'start_time':   datetime.now().isoformat(),
    }

    global_step         = 0
    current_epoch       = 1
    chunk_within_epoch  = 0
    total_training_time = 0.0

    cp = ckpt_mgr.load()
    if cp:
        print('\nREPRISE')
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(cp['model_state_dict'])
        # Supporte l'ancien format (optimizer_state_dict) et le nouveau (muon+adamw)
        if 'muon_state_dict' in cp and 'adamw_state_dict' in cp:
            muon_opt.load_state_dict(cp['muon_state_dict'])
            adamw_opt.load_state_dict(cp['adamw_state_dict'])
        elif 'optimizer_state_dict' in cp:
            print('  WARN : checkpoint ancien format (AdamW seul) — Muon repart de zéro')
            adamw_opt.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        # Resync lr après reprise
        for pg in muon_opt.param_groups:
            pg['lr'] = scheduler.get_lr() * 5.0
        for pg in adamw_opt.param_groups:
            pg['lr'] = scheduler.get_lr()

        current_epoch       = cp.get('current_epoch', 1)
        chunk_within_epoch  = cp.get('chunk_within_epoch', 0)
        global_step         = cp.get('global_step', 0)
        total_training_time = cp.get('total_training_time', 0.0)
        training_history    = cp.get('training_history', training_history)
        print(f'  -> epoch={current_epoch}  cwi={chunk_within_epoch}  '
              f'step={global_step:,}')

        if current_epoch > CONFIG['num_epochs']:
            print(f'\n✅ Training déjà terminé ({CONFIG["num_epochs"]} epochs).')
            print(f'   Pour continuer : augmente num_epochs dans CONFIG et relance.')
            print(f'   Checkpoint : {ckpt_mgr.path}')
            return

    print('\n' + '='*80)
    print(f'TRAINING START')
    print(f'  epochs {current_epoch} → {CONFIG["num_epochs"]}  |  '
          f'chunks/epoch={CONFIG["chunks_per_epoch"]}  |  '
          f'1 chunk en RAM à la fois  |  sliding window')
    print('='*80)

    for epoch in range(current_epoch, CONFIG['num_epochs'] + 1):

        ep_start  = (epoch - 1) * CONFIG['chunks_per_epoch']
        ep_chunks = ALL_TRAIN_CHUNKS[ep_start:ep_start + CONFIG['chunks_per_epoch']]
        start_cwi = chunk_within_epoch if epoch == current_epoch else 0

        chunk_ids_str = ' '.join(f'chunk_{c["id"]:03d}' for c in ep_chunks)
        print(f'\nEPOCH {epoch}/{CONFIG["num_epochs"]} — chunks : {chunk_ids_str}')
        if start_cwi > 0:
            print(f'  (reprise à cwi={start_cwi})')

        for cwi in range(start_cwi, CONFIG['chunks_per_epoch']):
            chunk_info = ep_chunks[cwi]

            try:
                global_step, total_training_time = train_one_chunk(
                    model               = model,
                    chunk_info          = chunk_info,
                    optimizers          = optimizers,
                    scheduler           = scheduler,
                    checkpoint_manager  = ckpt_mgr,
                    training_history    = training_history,
                    global_step         = global_step,
                    total_training_time = total_training_time,
                    current_epoch       = epoch,
                    chunk_within_epoch  = cwi,
                )

            except KeyboardInterrupt:
                print('\nCTRL+C — sauvegarde...')
                ckpt_mgr.save(model, optimizers, scheduler, metadata={
                    'current_epoch':       epoch,
                    'chunk_within_epoch':  cwi,
                    'global_step':         global_step,
                    'total_training_time': total_training_time,
                    'training_history':    training_history,
                })
                return

            except Exception:
                print(f'\nERREUR chunk_{chunk_info["id"]:03d} :\n{traceback.format_exc()}')
                ckpt_mgr.save(model, optimizers, scheduler, metadata={
                    'current_epoch':       epoch,
                    'chunk_within_epoch':  cwi,
                    'global_step':         global_step,
                    'total_training_time': total_training_time,
                    'training_history':    training_history,
                })
                raise

            next_cwi = cwi + 1
            if next_cwi >= CONFIG['chunks_per_epoch']:
                save_ep, save_cwi = epoch + 1, 0
            else:
                save_ep, save_cwi = epoch, next_cwi

            pct = int(next_cwi / CONFIG['chunks_per_epoch'] * 100)
            print(f'\n  [{pct}% epoch {epoch}] Save 25%...')
            ckpt_mgr.save(model, optimizers, scheduler, metadata={
                'current_epoch':       save_ep,
                'chunk_within_epoch':  save_cwi,
                'global_step':         global_step,
                'total_training_time': total_training_time,
                'training_history':    training_history,
            })

        ep_hist  = [c for c in training_history['chunks']
                    if c['current_epoch'] == epoch]
        avg_loss = sum(c['train_loss'] for c in ep_hist) / max(len(ep_hist), 1)

        print(f'\n{"="*80}')
        print(f'EPOCH {epoch} TERMINÉE  loss={avg_loss:.4f}  '
              f'step={global_step:,}  time={total_training_time/3600:.2f}h')
        print(f'{"="*80}')

        training_history['epochs'].append({
            'epoch':       epoch,
            'train_loss':  avg_loss,
            'global_step': global_step,
            'time_sec':    total_training_time,
        })

        chunk_within_epoch = 0

    print(f'\n{"="*80}\nTRAINING TERMINÉ\n{"="*80}')
    print(f'  Epochs  : {CONFIG["num_epochs"]}')
    print(f'  Steps   : {global_step:,}')
    print(f'  Temps   : {total_training_time/3600:.2f}h')
    if training_history.get('validations'):
        last = training_history['validations'][-1]
        print(f'  Val PPL : {last["val_ppl"]:.2f}  Val Loss : {last["val_loss"]:.4f}')
    print(f'  Checkpoint : {ckpt_mgr.path}')

    ckpt_mgr.save(model, optimizers, scheduler, metadata={
        'current_epoch':       CONFIG['num_epochs'] + 1,
        'chunk_within_epoch':  0,
        'global_step':         global_step,
        'total_training_time': total_training_time,
        'training_history':    training_history,
    })

    history_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f'  History : {history_path}')
    save_graphs(training_history)
    print('DONE')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())
    finally:
        print('\nBye')
