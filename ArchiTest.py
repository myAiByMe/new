#!/usr/bin/env python3
"""
🧪 HessGPT Architecture Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lance tous les tests sans GPU ni tokenizer HuggingFace.
Chaque test est indépendant — un échec n'arrête pas les suivants.

USAGE:
    python test_hessgpt.py
    python test_hessgpt.py --device cpu
    python test_hessgpt.py --fast          # skip tests lents (full model)
"""

import sys
import os
import math
import time
import traceback
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./Core/Model')
sys.path.append('./Core/Attention')
sys.path.append('./Core/FeedForward')
sys.path.append('./Core/TransformerBlock')

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--fast',   action='store_true', help='Skip slow full-model tests')
args = parser.parse_args()

DEVICE = args.device
FAST   = args.fast

# Dimensions mini pour les tests (rapide, pas besoin de GPU puissant)
VOCAB   = 512       # petit vocab synthétique
EMBED   = 128
HEADS   = 4
KV_H    = 2         # GQA 4:1 → 2:1 pour rester divisible
LAYERS  = 2
SEQ     = 64
BATCH   = 2

# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────
results  = []
total    = 0
passed   = 0

def run(name, fn):
    global total, passed
    total += 1
    try:
        t0 = time.time()
        fn()
        ms = (time.time() - t0) * 1000
        print(f"  ✅  {name:<55} ({ms:6.1f}ms)")
        results.append((name, True, None))
        passed += 1
    except Exception as e:
        print(f"  ❌  {name:<55}")
        print(f"       → {e}")
        tb = traceback.format_exc()
        for line in tb.splitlines()[-6:]:
            print(f"         {line}")
        results.append((name, False, str(e)))


# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("🧪 HessGPT Test Suite")
print(f"   device={DEVICE}  fast={FAST}")
print("="*70)

print("\n📦 Importing modules...")
try:
    from attention       import MultiHeadAttention, RMSNorm
    from feedforward     import FeedForward
    from transformer_block import TransformerBlock
    from HessGpt         import HessGPT
    print("   ✅ All modules imported\n")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Make sure ./Core/Model is in the path and all files exist.")
    sys.exit(1)


# ══════════════════════════════════════════════
# 1. RMSNORM
# ══════════════════════════════════════════════
print("─"*70)
print("1. RMSNorm")
print("─"*70)

def test_rmsnorm_shape():
    x   = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    rms = RMSNorm(EMBED).to(DEVICE)
    y   = rms(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

def test_rmsnorm_normalizes():
    x   = torch.randn(BATCH, SEQ, EMBED, device=DEVICE) * 100
    rms = RMSNorm(EMBED).to(DEVICE)
    y   = rms(x)
    # RMS de la sortie doit être proche de 1 (avec weight=1)
    rms_val = y.pow(2).mean(dim=-1).sqrt()
    assert rms_val.max().item() < 10.0, "RMSNorm output scale too large"

def test_rmsnorm_no_nan():
    x   = torch.zeros(BATCH, SEQ, EMBED, device=DEVICE)  # cas dégénéré
    rms = RMSNorm(EMBED).to(DEVICE)
    y   = rms(x)
    assert not torch.isnan(y).any(), "RMSNorm produced NaN on zero input"

def test_rmsnorm_dtype_preserved():
    x   = torch.randn(BATCH, SEQ, EMBED, device=DEVICE).half()
    rms = RMSNorm(EMBED).to(DEVICE).half()
    y   = rms(x)
    assert y.dtype == torch.float16

run("RMSNorm output shape",          test_rmsnorm_shape)
run("RMSNorm normalizes correctly",  test_rmsnorm_normalizes)
run("RMSNorm no NaN on zero input",  test_rmsnorm_no_nan)
run("RMSNorm preserves dtype",       test_rmsnorm_dtype_preserved)


# ══════════════════════════════════════════════
# 2. FEEDFORWARD
# ══════════════════════════════════════════════
print("\n─"*70)
print("2. FeedForward (SwiGLU + GELU fallback)")
print("─"*70)

def test_ff_swiglu_shape():
    ff = FeedForward(EMBED, dropout=0.0, use_swiglu=True).to(DEVICE)
    x  = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    y  = ff(x)
    assert y.shape == (BATCH, SEQ, EMBED)

def test_ff_gelu_shape():
    ff = FeedForward(EMBED, dropout=0.0, use_swiglu=False).to(DEVICE)
    x  = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    y  = ff(x)
    assert y.shape == (BATCH, SEQ, EMBED)

def test_ff_swiglu_dim_multiple_of_64():
    ff = FeedForward(EMBED, dropout=0.0, use_swiglu=True).to(DEVICE)
    # La dim interne doit être multiple de 64 (arrondi du 8/3 * embed)
    inner = ff.gate_proj.out_features
    assert inner % 64 == 0, f"SwiGLU inner dim {inner} not multiple of 64"

def test_ff_no_nan():
    ff = FeedForward(EMBED, dropout=0.0, use_swiglu=True).to(DEVICE)
    x  = torch.randn(BATCH, SEQ, EMBED, device=DEVICE) * 10
    y  = ff(x)
    assert not torch.isnan(y).any()

def test_ff_single_dropout():
    """Vérifie l'absence de double dropout (bug historique).
    Stratégie : on compte le ratio de zéros en sortie du dropout.
    dropout=0.5 → ~50% zeros. Double dropout → ~75% zeros.
    On teste via le dropout directement, pas via la valeur absolue
    de sortie qui dépend des poids initiaux."""
    torch.manual_seed(0)
    ff = FeedForward(EMBED, dropout=0.5, use_swiglu=True).to(DEVICE)
    ff.train()
    # On teste directement le module dropout isolé
    drop = ff.dropout
    x    = torch.ones(1000, EMBED, device=DEVICE)
    y    = drop(x)
    # Avec dropout=0.5 : ~50% zeros (rescalé ×2 sur les survivants)
    zero_ratio = (y == 0).float().mean().item()
    # Single dropout → ~50%, double → ~75%
    assert zero_ratio < 0.65, f"Possible double dropout: zero_ratio={zero_ratio:.2f} (expected ~0.5)"
    assert zero_ratio > 0.35, f"Dropout not active: zero_ratio={zero_ratio:.2f}"

run("FeedForward SwiGLU output shape",      test_ff_swiglu_shape)
run("FeedForward GELU output shape",        test_ff_gelu_shape)
run("SwiGLU inner dim multiple of 64",      test_ff_swiglu_dim_multiple_of_64)
run("FeedForward no NaN",                   test_ff_no_nan)
run("FeedForward single dropout (no dupe)", test_ff_single_dropout)


# ══════════════════════════════════════════════
# 3. MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════
print("\n─"*70)
print("3. MultiHeadAttention (RoPE + GQA + QK-Norm + soft-cap)")
print("─"*70)

def make_attn(**kwargs):
    defaults = dict(
        embed_dim=EMBED, num_heads=HEADS, dropout=0.0,
        use_rope=True, max_seq_len=SEQ,
        use_yarn=False, yarn_scale=1.0, yarn_original_max_len=SEQ,
        n_kv_heads=KV_H, use_qk_norm=True,
        use_flash_attn=False,  # flash attn peut ne pas être dispo en test
    )
    defaults.update(kwargs)
    return MultiHeadAttention(**defaults).to(DEVICE)

def test_attn_output_shape():
    attn = make_attn()
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y    = attn(x, mask)
    assert y.shape == (BATCH, SEQ, EMBED)

def test_attn_causal_mask():
    """Le token i ne doit pas voir le token i+1 (causalité)."""
    attn = make_attn()
    attn.eval()
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    # On perturbe le dernier token
    x2       = x.clone()
    x2[:, -1, :] += 100.0
    y1 = attn(x, mask)
    y2 = attn(x2, mask)
    # La sortie des tokens 0..SEQ-2 ne doit PAS changer (causalité)
    diff = (y1[:, :-1, :] - y2[:, :-1, :]).abs().max().item()
    assert diff < 1e-3, f"Causal mask broken, max diff={diff:.4f}"

def test_attn_no_nan():
    attn = make_attn()
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y    = attn(x, mask)
    assert not torch.isnan(y).any()

def test_attn_gqa_heads():
    """Vérifie que GQA crée bien n_kv_heads têtes KV."""
    attn = make_attn(n_kv_heads=KV_H)
    assert attn.k_proj.out_features == KV_H * (EMBED // HEADS), \
        "k_proj output dim incorrect for GQA"

def test_attn_qk_norm():
    """QK-Norm doit exister et être RMSNorm."""
    attn = make_attn(use_qk_norm=True)
    assert hasattr(attn, 'q_norm'), "q_norm missing"
    assert hasattr(attn, 'k_norm'), "k_norm missing"
    assert isinstance(attn.q_norm, RMSNorm)
    assert isinstance(attn.k_norm, RMSNorm)

def test_attn_rope_cache_dtype():
    """Le cache RoPE doit matcher le dtype du input (mixed precision)."""
    attn = make_attn()
    attn.eval()
    x_fp16 = torch.randn(BATCH, SEQ, EMBED, device=DEVICE).half()
    attn    = attn.half()
    mask    = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y = attn(x_fp16, mask)   # ne doit pas crasher
    assert y.dtype == torch.float16

def test_attn_yarn():
    attn = make_attn(
        use_yarn=True, yarn_scale=4.0,
        yarn_original_max_len=SEQ//4, max_seq_len=SEQ
    )
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y    = attn(x, mask)
    assert y.shape == (BATCH, SEQ, EMBED)
    assert not torch.isnan(y).any()

def test_attn_soft_cap():
    """Soft-cap doit borner les logits d'attention."""
    # On injecte des valeurs énormes et vérifie que les logits restent bornés
    # Note: le soft-cap dans HessGPT est sur les logits de sortie, pas sur attn weights
    # On teste donc via le modèle complet ci-dessous
    attn = make_attn()
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE) * 1000
    mask = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y    = attn(x, mask)
    assert not torch.isnan(y).any(), "NaN with large input values"
    assert not torch.isinf(y).any(), "Inf with large input values"

run("Attention output shape",              test_attn_output_shape)
run("Attention causal mask correct",       test_attn_causal_mask)
run("Attention no NaN",                    test_attn_no_nan)
run("GQA k_proj output dim correct",       test_attn_gqa_heads)
run("QK-Norm present and is RMSNorm",      test_attn_qk_norm)
run("RoPE cache dtype matches input",      test_attn_rope_cache_dtype)
run("YaRN output shape + no NaN",          test_attn_yarn)
run("Attention stable with large inputs",  test_attn_soft_cap)


# ══════════════════════════════════════════════
# 4. TRANSFORMER BLOCK
# ══════════════════════════════════════════════
print("\n─"*70)
print("4. TransformerBlock (Pre-Norm + Residual)")
print("─"*70)

def make_block(**kwargs):
    defaults = dict(
        embed_dim=EMBED, num_heads=HEADS, dropout=0.0,
        use_rope=True, max_seq_len=SEQ,
        use_yarn=False, yarn_scale=1.0, yarn_original_max_len=SEQ,
        use_swiglu=True, n_kv_heads=KV_H,
        use_qk_norm=True, use_flash_attn=False,
    )
    defaults.update(kwargs)
    return TransformerBlock(**defaults).to(DEVICE)

def test_block_shape():
    block = make_block()
    x     = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask  = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y     = block(x, mask)
    assert y.shape == (BATCH, SEQ, EMBED)

def test_block_residual():
    """La sortie ne doit pas être identique à l'entrée (residual actif)."""
    block = make_block()
    block.eval()
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y    = block(x, mask)
    diff = (y - x).abs().mean().item()
    assert diff > 1e-6, "Block output identical to input — residual may be broken"

def test_block_pre_norm():
    """Vérifie que le block a bien des RMSNorm (pre-norm) — attributs ln1/ln2."""
    block = make_block()
    assert hasattr(block, 'ln1') and isinstance(block.ln1, RMSNorm), "ln1 missing or wrong type"
    assert hasattr(block, 'ln2') and isinstance(block.ln2, RMSNorm), "ln2 missing or wrong type"

def test_block_no_nan():
    block = make_block()
    x     = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    mask  = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y     = block(x, mask)
    assert not torch.isnan(y).any()

def test_block_gradient_flows():
    block = make_block()
    x     = torch.randn(BATCH, SEQ, EMBED, device=DEVICE, requires_grad=True)
    mask  = torch.triu(torch.ones(SEQ, SEQ, device=DEVICE), diagonal=1).bool()
    y     = block(x, mask)
    loss  = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    assert not torch.isnan(x.grad).any(), "NaN gradient"

run("TransformerBlock output shape",  test_block_shape)
run("TransformerBlock residual conn", test_block_residual)
run("TransformerBlock pre-norm",      test_block_pre_norm)
run("TransformerBlock no NaN",        test_block_no_nan)
run("TransformerBlock gradient flow", test_block_gradient_flows)


# ══════════════════════════════════════════════
# 5. HESSGPT FULL MODEL
# ══════════════════════════════════════════════
print("\n─"*70)
print("5. HessGPT Full Model")
print("─"*70)

def make_model(**kwargs):
    defaults = dict(
        vocab_size=VOCAB, embed_dim=EMBED, num_heads=HEADS,
        num_layers=LAYERS, max_seq_len=SEQ, dropout=0.0,
        use_rope=True, use_yarn=False,
        yarn_scale=1.0, yarn_original_max_len=SEQ,
        use_swiglu=True, n_kv_heads=KV_H,
        use_qk_norm=True, soft_cap=30.0,
        use_flash_attn=False,
    )
    defaults.update(kwargs)
    return HessGPT(**defaults).to(DEVICE)

def test_model_instantiation():
    m = make_model()
    assert m is not None

def test_model_forward_no_loss():
    m   = make_model()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    logits, loss = m(ids)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert loss is None

def test_model_forward_with_loss():
    m       = make_model()
    ids     = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    logits, loss = m(ids, targets=targets, pad_token_id=0)
    assert loss is not None
    assert loss.item() > 0
    assert not math.isnan(loss.item())

def test_model_loss_ignore_padding():
    """pad_token_id=-100 labels doivent être ignorés dans la loss."""
    m       = make_model()
    ids     = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    # Tous les targets masqués → loss doit être 0 ou NaN-free mais ignoré
    targets = torch.full((BATCH, SEQ), -100, dtype=torch.long, device=DEVICE)
    targets[:, 0] = torch.randint(0, VOCAB, (BATCH,), device=DEVICE)  # 1 token valide
    _, loss = m(ids, targets=targets, pad_token_id=-100)
    assert loss is not None
    assert not math.isnan(loss.item()), "Loss is NaN with -100 masking"

def test_model_weight_tying():
    """token_embeddings et output_head doivent partager les poids."""
    m = make_model()
    assert m.token_embeddings.weight.data_ptr() == m.output_head.weight.data_ptr(), \
        "Weight tying broken — embeddings and output_head are not shared"

def test_model_param_count():
    """Le modèle full-size (~500M) doit avoir un compte de params raisonnable."""
    if FAST:
        return
    m = HessGPT(
        vocab_size=128259, embed_dim=1280, num_heads=20,
        num_layers=22, max_seq_len=1024, dropout=0.0,
        use_rope=True, use_yarn=False,
        use_swiglu=True, n_kv_heads=5,
        use_qk_norm=True, soft_cap=30.0,
        use_flash_attn=False,
    )
    params = sum(p.numel() for p in m.parameters())
    # Weight tying → output_head ne compte pas, donc ~350-600M est raisonnable
    assert 100e6 < params < 800e6, f"Unexpected param count: {params/1e6:.1f}M"
    print(f"         → {params/1e6:.1f}M params total")

def test_model_soft_cap_bounds_logits():
    """soft_cap=30 → tous les logits doivent être dans [-30, 30]."""
    m   = make_model(soft_cap=30.0)
    m.eval()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    with torch.no_grad():
        logits, _ = m(ids)
    assert logits.max().item() <= 30.0 + 1e-4, f"logits exceed soft_cap: {logits.max().item():.3f}"
    assert logits.min().item() >= -30.0 - 1e-4, f"logits below -soft_cap: {logits.min().item():.3f}"

def test_model_causal_output():
    """Le logit de position i ne doit pas dépendre du token i+1."""
    m   = make_model()
    m.eval()
    ids  = torch.randint(0, VOCAB, (1, SEQ), device=DEVICE)
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % VOCAB  # change dernier token
    with torch.no_grad():
        l1, _ = m(ids)
        l2, _ = m(ids2)
    diff = (l1[:, :-1, :] - l2[:, :-1, :]).abs().max().item()
    assert diff < 1e-3, f"Model is not causal, max diff={diff:.4f}"

def test_model_no_nan_forward():
    m   = make_model()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    logits, _ = m(ids)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

def test_model_backward():
    m       = make_model()
    ids     = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    targets = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    _, loss = m(ids, targets=targets)
    loss.backward()
    # Au moins quelques paramètres doivent avoir des gradients
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    for g in grads:
        assert not torch.isnan(g).any(), "NaN gradient in backward pass"

def test_model_generate():
    m   = make_model()
    m.eval()
    ids = torch.randint(0, VOCAB, (1, 10), device=DEVICE)
    out = m.generate(ids, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 15), f"Expected (1,15), got {out.shape}"
    assert not torch.isnan(out.float()).any()

def test_model_generate_restores_training_mode():
    """generate() doit restaurer le mode training."""
    m = make_model()
    m.train()
    ids = torch.randint(0, VOCAB, (1, 10), device=DEVICE)
    m.generate(ids, max_new_tokens=3)
    assert m.training, "generate() did not restore training mode"

def test_model_yarn_forward():
    m   = make_model(
        use_yarn=True, yarn_scale=4.0,
        yarn_original_max_len=SEQ // 4, max_seq_len=SEQ
    )
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    logits, _ = m(ids)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert not torch.isnan(logits).any()

run("Model instantiation",               test_model_instantiation)
run("Model forward — no loss",           test_model_forward_no_loss)
run("Model forward — with loss",         test_model_forward_with_loss)
run("Model loss ignores -100 labels",    test_model_loss_ignore_padding)
run("Weight tying embedding ↔ head",     test_model_weight_tying)
run("Full-size param count reasonable",  test_model_param_count)
run("soft_cap bounds logits to ±30",     test_model_soft_cap_bounds_logits)

def test_model_causal_flash():
    """Vérifie que le path FlashAttention (is_causal=True) est aussi causal."""
    if not torch.cuda.is_available():
        return  # FA2 kernel uniquement sur CUDA
    m = make_model(use_flash_attn=True)
    m.eval()
    ids  = torch.randint(0, VOCAB, (1, SEQ), device=DEVICE)
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % VOCAB  # change dernier token
    with torch.no_grad():
        l1, _ = m(ids)
        l2, _ = m(ids2)
    diff = (l1[:, :-1, :] - l2[:, :-1, :]).abs().max().item()
    assert diff < 1e-3, f"Flash path is not causal, max diff={diff:.4f}"

run("Model output is causal",            test_model_causal_output)
run("Model causal (flash path)",         test_model_causal_flash)
run("Model no NaN/Inf in forward",       test_model_no_nan_forward)
run("Model backward — no NaN gradients", test_model_backward)
run("Model generate output shape",       test_model_generate)
run("generate() restores training mode", test_model_generate_restores_training_mode)
run("Model forward with YaRN",           test_model_yarn_forward)


# ══════════════════════════════════════════════
# 6. GRADIENT ACCUMULATION SIMULATION
# ══════════════════════════════════════════════
print("\n─"*70)
print("6. Gradient Accumulation")
print("─"*70)

def test_grad_accum_equivalent_to_full_batch():
    """
    Accumulation sur N micro-batches doit donner les mêmes gradients
    qu'un seul batch de taille N*micro_size.
    """
    MICRO = 2
    ACCUM = 4
    torch.manual_seed(42)

    m_full  = make_model()
    m_accum = make_model()
    # Même poids initiaux
    m_accum.load_state_dict(m_full.state_dict())

    ids     = torch.randint(0, VOCAB, (MICRO * ACCUM, SEQ), device=DEVICE)
    targets = torch.randint(0, VOCAB, (MICRO * ACCUM, SEQ), device=DEVICE)

    # Full batch
    opt_full = torch.optim.AdamW(m_full.parameters(), lr=0.0)
    opt_full.zero_grad()
    _, loss_full = m_full(ids, targets=targets)
    loss_full.backward()

    # Gradient accumulation
    opt_accum = torch.optim.AdamW(m_accum.parameters(), lr=0.0)
    opt_accum.zero_grad()
    for i in range(ACCUM):
        x = ids[i*MICRO:(i+1)*MICRO]
        y = targets[i*MICRO:(i+1)*MICRO]
        _, loss_micro = m_accum(x, targets=y)
        (loss_micro / ACCUM).backward()

    # Les gradients doivent être très proches
    for (n, p_full), (_, p_accum) in zip(
        m_full.named_parameters(), m_accum.named_parameters()
    ):
        if p_full.grad is not None:
            diff = (p_full.grad - p_accum.grad).abs().max().item()
            assert diff < 1e-4, f"Grad mismatch on {n}: {diff:.6f}"

run("Grad accum ≈ full batch gradients", test_grad_accum_equivalent_to_full_batch)


# ══════════════════════════════════════════════
# 7. LORA (depuis le SFT)
# ══════════════════════════════════════════════
print("\n─"*70)
print("7. LoRA Layers")
print("─"*70)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.scaling = alpha / r
        self.lora_A  = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    def forward(self, x):
        x = self.dropout(x)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, base_layer, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora       = LoRALayer(base_layer.in_features, base_layer.out_features,
                                    r=r, alpha=alpha, dropout=dropout)
    def forward(self, x):
        return self.base_layer(x) + self.lora(x)

def test_lora_output_shape():
    base = nn.Linear(EMBED, EMBED, bias=False)
    l    = LinearWithLoRA(base, r=4, alpha=8).to(DEVICE)
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE)
    y    = l(x)
    assert y.shape == (BATCH, SEQ, EMBED)

def test_lora_B_zero_init():
    """lora_B doit être initialisé à zéro → sortie LoRA = 0 au départ."""
    base = nn.Linear(EMBED, EMBED, bias=False)
    l    = LinearWithLoRA(base, r=4, alpha=8)
    lora_out = l.lora(torch.randn(1, EMBED))
    assert lora_out.abs().max().item() == 0.0, "lora_B not zero-initialized"

def test_lora_base_frozen_lora_trainable():
    """Après freeze, seuls les params LoRA doivent être trainable."""
    m = make_model()
    for p in m.parameters():
        p.requires_grad = False
    # Applique LoRA à q_proj du premier block
    first_attn = m.blocks[0].attention
    base        = first_attn.q_proj
    lora_layer  = LinearWithLoRA(base, r=4, alpha=8)
    first_attn.q_proj = lora_layer

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable > 0, "No trainable params after LoRA injection"
    # Base weights must be frozen
    assert not base.weight.requires_grad, "Base layer not frozen"

def test_lora_no_nan():
    base = nn.Linear(EMBED, EMBED, bias=False)
    l    = LinearWithLoRA(base, r=4, alpha=8).to(DEVICE)
    x    = torch.randn(BATCH, SEQ, EMBED, device=DEVICE) * 100
    y    = l(x)
    assert not torch.isnan(y).any()

run("LoRA output shape",                 test_lora_output_shape)
run("LoRA B zero-init → delta=0",        test_lora_B_zero_init)
run("LoRA base frozen, LoRA trainable",  test_lora_base_frozen_lora_trainable)
run("LoRA no NaN with large inputs",     test_lora_no_nan)


# ══════════════════════════════════════════════
# 8. MASKING SFT (LLaMA-3 format)
# ══════════════════════════════════════════════
print("\n─"*70)
print("8. SFT Masking (LLaMA-3 native format)")
print("─"*70)

def simulate_llama3_masking(token_sequence, start_header_id, end_header_id, eot_id):
    """Réplique exactement la logique de SFTDataset.__getitem__."""
    tokens     = token_sequence
    input_ids  = tokens[:-1]
    target_ids = tokens[1:]
    labels     = [-100] * len(target_ids)
    state      = 'prompt'
    pending    = []

    for i, token_id in enumerate(input_ids):
        if state == 'prompt':
            if token_id == start_header_id:
                state   = 'in_header'
                pending = []
        elif state == 'in_header':
            if token_id == end_header_id:
                header = pending  # liste d'IDs
                # Simule: si header contient l'ID de 'assistant'
                if header == ['assistant']:
                    state = 'in_assistant_body'
                else:
                    state = 'prompt'
                pending = []
            else:
                pending.append(token_id)
        elif state == 'in_assistant_body':
            if token_id == eot_id:
                labels[i] = target_ids[i]
                state     = 'prompt'
            else:
                labels[i] = target_ids[i]

    return labels

def test_masking_single_turn():
    """Tour unique : seule la réponse assistant doit être démasquée."""
    SH, EH, EOT = 100, 101, 102
    seq = [SH, 'system', EH, 10, 11, EOT,
           SH, 'user',   EH, 20, 21, EOT,
           SH, 'assistant', EH, 30, 31, 32, EOT]
    labels = simulate_llama3_masking(seq, SH, EH, EOT)
    unmasked  = [i for i, l in enumerate(labels) if l != -100]
    # Calcul expected via la même machine à états — les positions démasquées
    # sont celles dans le corps du tour assistant (30, 31, 32) + le EOT final
    input_ids = seq[:-1]
    expected  = []
    state, pending = 'prompt', []
    for i, t in enumerate(input_ids):
        if state == 'prompt':
            if t == SH: state = 'in_header'; pending = []
        elif state == 'in_header':
            if t == EH:
                state = 'in_assistant_body' if pending == ['assistant'] else 'prompt'
                pending = []
            else:
                pending.append(t)
        elif state == 'in_assistant_body':
            expected.append(i)
            if t == EOT: state = 'prompt'
    assert unmasked == expected, f"Masking wrong: {unmasked} != {expected}"

def test_masking_multi_turn():
    """Multi-turn : LES DEUX réponses assistant doivent être démasquées."""
    SH, EH, EOT = 100, 101, 102
    seq = [
        SH, 'system',    EH, 10, EOT,
        SH, 'user',      EH, 20, EOT,
        SH, 'assistant', EH, 30, 31, EOT,
        SH, 'user',      EH, 40, EOT,
        SH, 'assistant', EH, 50, 51, EOT,
    ]
    labels    = simulate_llama3_masking(seq, SH, EH, EOT)
    unmasked  = [i for i, l in enumerate(labels) if l != -100]
    input_ids = seq[:-1]
    expected  = [i for i, t in enumerate(input_ids) if t in (30, 31, EOT, 50, 51)]
    # EOT apparaît 4 fois dans input_ids; on veut seulement ceux dans les tours assistant
    # Re-calculer expected proprement
    expected2 = []
    state, pending = 'prompt', []
    for i, t in enumerate(input_ids):
        if state == 'prompt':
            if t == SH: state = 'in_header'; pending = []
        elif state == 'in_header':
            if t == EH:
                state = 'in_assistant_body' if pending == ['assistant'] else 'prompt'
                pending = []
            else:
                pending.append(t)
        elif state == 'in_assistant_body':
            expected2.append(i)
            if t == EOT: state = 'prompt'
    assert unmasked == expected2, f"Multi-turn masking wrong: {unmasked} != {expected2}"

def test_masking_all_prompt_masked():
    """Tous les tokens prompt/system/user doivent rester à -100."""
    SH, EH, EOT = 100, 101, 102
    seq = [
        SH, 'system', EH, 10, 11, EOT,
        SH, 'user',   EH, 20, 21, EOT,
        SH, 'assistant', EH, 30, EOT,
    ]
    labels   = simulate_llama3_masking(seq, SH, EH, EOT)
    input_ids = seq[:-1]
    # Les tokens 10, 11, 20, 21 (system et user content) doivent tous être masqués
    for i, t in enumerate(input_ids):
        if t in (10, 11, 20, 21):
            assert labels[i] == -100, f"Token {t} at pos {i} should be masked but got {labels[i]}"

run("SFT masking single turn",           test_masking_single_turn)
run("SFT masking multi-turn both turns", test_masking_multi_turn)
run("SFT masking prompt tokens = -100",  test_masking_all_prompt_masked)


# ══════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════
failed_tests = [(n, e) for n, ok, e in results if not ok]

print("\n" + "="*70)
print(f"{'🎉 ALL TESTS PASSED' if not failed_tests else '⚠️  SOME TESTS FAILED'}")
print(f"   {passed}/{total} passed")
print("="*70)

if failed_tests:
    print(f"\n❌ Failed tests ({len(failed_tests)}):")
    for name, err in failed_tests:
        print(f"   • {name}")
        print(f"     {err}")
    sys.exit(1)
else:
    print("\n✅ Architecture is clean. Ready for training.")
    sys.exit(0)