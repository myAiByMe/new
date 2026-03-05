#!/usr/bin/env python3
"""
Ultra-Filtered Dataset Downloader — LLaMA-3 Tokenizer

DATASETS (mix 2B tokens/chunk):
  DCLM-Baseline     37% → mlfoundations/dclm-baseline-1.0
  FineWeb-Edu       22% → HuggingFaceFW/fineweb-edu (sample-100BT, int_score>=4, EN only)
  Zyda-2            18% → Zyphra/Zyda-2
  Cosmopedia v2     10% → HuggingFaceTB/cosmopedia-v2
  peS2o              5% → allenai/peS2o (v2, papiers scientifiques)
  FineWiki           5% → HuggingFaceFW/fineweb (CC-MAIN-2024-10, EN only)
  FineMath-4+        3% → HuggingFaceTB/finemath (finemath-4plus, no general filter)

FILTRE ANGLAIS PAR DATASET:
  - fineweb_edu / finewiki  → champ 'language' natif (standard HF)
  - dclm_baseline           → champ 'language_id_whole_page_fasttext' (score EN >= 0.65)
  - zyda2 / cosmopedia_v2   → English-only par design (tagué officiellement), pas de champ
  - pes2o / finemath_4plus  → English-only par design, pas de champ

USAGE:
  python downloader.py --num-chunks 5
"""

import os
import sys
import json
import hashlib
import signal
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import re
import time
from typing import Dict, List, Optional
import numpy as np

try:
    import zstandard
except ImportError:
    print("📦 Installing zstandard...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard", "-q"])
    import zstandard
    print("✅ zstandard installed!")

from huggingface_hub import login
login(token="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")  # ← remplace par ton token HF valide

# ============================================
# TIMEOUT HANDLER
# ============================================
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("⏰ Timeout atteint!")

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'output_dir':       './data/ultra_filtered',
    'checkpoint_dir':   './temp_checkpoints',
    'offsets_file':     './data/ultra_filtered/dataset_offsets.json',
    'tokenizer_name':   'NousResearch/Meta-Llama-3-8B',
    'num_chunks':       5,

    'checkpoint_interval': 100_000_000,  # 100M tokens
    'token_tolerance':     5_000_000,    # ±5M tokens
    'dataset_timeout':     13800 + 60*10,        # 3h50 par dataset

    # Filtres généraux (appliqués à TOUS les datasets sauf skip_general_filter=True)
    'min_text_length':           500,
    'max_text_length':           100000,
    'min_alpha_ratio':           0.7,
    'max_special_chars_ratio':   0.15,
    'min_avg_word_length':       3.0,
    'max_avg_word_length':       12.0,
    'min_unique_words_ratio':    0.4,
    'max_line_repetition_ratio': 0.3,

    # Seuil fasttext pour DCLM (score anglais minimum)
    'dclm_fasttext_en_threshold': 0.65,

    'enable_dedup': True,
}

# ============================================
# DATASETS — Mix 2B tokens/chunk
# ============================================
# lang_filter_mode:
#   'field'    → filtre via champ 'language' (FineWeb/FineWeb-Edu)
#   'fasttext' → filtre via champ 'language_id_whole_page_fasttext' (DCLM)
#   'none'     → dataset English-only par design, pas de filtre nécessaire
# ============================================
DATASETS = [

    {
        'name': 'finemath_4plus',
        'source': 'HuggingFaceTB/finemath',
        'config': 'finemath-4plus',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🧮 FineMath-4+ logique/raisonnement 3%',
        'tokens_per_chunk': 60_000_000,
        'int_score_min': 0,
        'lang_filter_mode': 'none',    # Math/logique, English-only par design
        'skip_general_filter': True,   # bypass filtre code/math — c'est le but
    },
    {
        'name': 'finewiki',
        'source': 'HuggingFaceFW/fineweb',
        'config': 'CC-MAIN-2024-10',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '📖 FineWiki CC-MAIN-2024-10 5% (English only)',
        'tokens_per_chunk': 100_000_000,
        'int_score_min': 0,
        'lang_filter_mode': 'field',   # champ 'language' natif HF
        'skip_general_filter': False,
    },
    {
        'name': 'pes2o',
        'source': 'allenai/olmo-mix-1124',  # ← changé
        'config': 'pes2o',                  # ← changé
        'split': 'train',
        'text_key': 'text',                 # ✅ inchangé
        'streaming': True,
        'description': '🔬 peS2o papiers scientifiques 5%',
        'tokens_per_chunk': 100_000_000,
        'int_score_min': 0,
        'lang_filter_mode': 'none',
        'skip_general_filter': True,
    },
    {
        'name': 'cosmopedia_v2',
        'source': 'HuggingFaceTB/cosmopedia-v2',
        'config': 'cosmopedia-v2',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🌌 Cosmopedia v2 textbook 10%',
        'tokens_per_chunk': 200_000_000,
        'int_score_min': 0,
        'lang_filter_mode': 'none',    # Synthétique, English-only par design
        'skip_general_filter': False,
    },
    {
        'name': 'zyda2',
        'source': 'Zyphra/Zyda-2',
        'config': 'default',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '⚡ Zyda-2 18%',
        'tokens_per_chunk': 360_000_000,
        'int_score_min': 0,
        'lang_filter_mode': 'none',    # English-only par design (officiel)
        'skip_general_filter': False,
    },
    {
        'name': 'fineweb_edu',
        'source': 'HuggingFaceFW/fineweb-edu',
        'config': 'sample-100BT',
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🎓 FineWeb-Edu int_score>=4 22%',
        'tokens_per_chunk': 440_000_000,
        'int_score_min': 4,
        'lang_filter_mode': 'field',   # champ 'language' natif HF
        'skip_general_filter': False,
    },
    {
        'name': 'dclm_baseline',
        'source': 'mlfoundations/dclm-baseline-1.0',
        'config': None,
        'split': 'train',
        'text_key': 'text',
        'streaming': True,
        'description': '🌍 DCLM-Baseline 37%',
        'tokens_per_chunk': 740_000_000,
        'int_score_min': 0,
        'lang_filter_mode': 'fasttext', # champ 'language_id_whole_page_fasttext'
        'skip_general_filter': False,
    },
]

# ============================================
# TOKENS SPÉCIAUX — LLAMA-3
# ============================================
SPECIAL_TOKENS = [
    '<code>',
    '<think>',
    '</think>',
]

# ============================================
# PATTERNS CODE / MATH
# ============================================
CODE_PATTERNS = [
    r'\bdef\s+\w+\s*\(', r'\bfunction\s+\w+\s*\(', r'\bclass\s+\w+\s*[:{]',
    r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import', r'#include\s*<',
    r'\bpublic\s+static\s+void', r'\bprivate\s+\w+\s+\w+\s*\(',
    r'=>\s*{', r'\bconst\s+\w+\s*=', r'\bvar\s+\w+\s*=', r'\blet\s+\w+\s*=',
    r'SELECT\s+.+FROM', r'INSERT\s+INTO', r'CREATE\s+TABLE',
    r'```\w+', r'<\?php', r'<script>', r'</script>',
    r'System\.out\.println', r'console\.log', r'printf\s*\(',
]

MATH_PATTERNS = [
    r'\\begin\{equation\}', r'\\frac\{', r'\\sum_', r'\\int_', r'\\sqrt\{',
    r'\$\$', r'\\alpha|\\beta|\\gamma|\\delta',
    r'Theorem\s+\d+', r'Lemma\s+\d+', r'Proof\.', r'Q\.E\.D\.',
    r'\b[a-z]\s*=\s*[a-z]\s*\+\s*[a-z]\b', r'\bf\(x\)\s*=', r'\d+x\s*[\+\-]\s*\d+',
]

CODE_REGEX = [re.compile(p, re.IGNORECASE) for p in CODE_PATTERNS]
MATH_REGEX = [re.compile(p, re.IGNORECASE) for p in MATH_PATTERNS]

def contains_code_or_math(text: str) -> bool:
    for pattern in CODE_REGEX:
        if pattern.search(text):
            return True
    for pattern in MATH_REGEX:
        if pattern.search(text):
            return True
    return False

# ============================================
# FILTRES LANGUE — 3 modes
# ============================================

def is_english_field(doc: dict) -> bool:
    """
    Mode 'field' : champ 'language' standard (FineWeb, FineWeb-Edu).
    Retourne True si EN ou si champ absent.
    """
    lang = doc.get('language', None)
    if lang is None:
        return True
    return lang.lower().startswith('en')

def is_english_fasttext(doc: dict) -> bool:
    """
    Mode 'fasttext' : champ 'language_id_whole_page_fasttext' (DCLM-Baseline).
    Structure attendue: {'en': 0.92, ...}
    Retourne True si score EN >= seuil ou si champ absent.
    """
    fasttext = doc.get('language_id_whole_page_fasttext', None)
    if fasttext is None:
        return True  # pas de champ → on garde (DCLM est ~99% EN de toute façon)
    if isinstance(fasttext, dict):
        return fasttext.get('en', 0.0) >= CONFIG['dclm_fasttext_en_threshold']
    return True

def is_english(doc: dict, mode: str) -> bool:
    """
    Dispatch vers le bon filtre selon le mode du dataset.
    """
    if mode == 'field':
        return is_english_field(doc)
    elif mode == 'fasttext':
        return is_english_fasttext(doc)
    else:  # 'none' — dataset English-only par design
        return True

# ============================================
# FILTRE GÉNÉRAL
# ============================================
def filter_document(text: str) -> bool:
    if len(text) < CONFIG['min_text_length'] or len(text) > CONFIG['max_text_length']:
        return False
    if contains_code_or_math(text):
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / len(text) < CONFIG['min_alpha_ratio']:
        return False
    special_chars = sum(not c.isalnum() and not c.isspace() for c in text)
    if special_chars / len(text) > CONFIG['max_special_chars_ratio']:
        return False
    if text.count('http') > 3 or text.count('www.') > 3:
        return False
    spam_patterns = [
        'click here', 'buy now', 'subscribe', 'follow us',
        'copyright ©', 'all rights reserved', 'terms of service',
        'cookies policy', 'privacy policy',
    ]
    text_lower = text.lower()
    if sum(text_lower.count(p) for p in spam_patterns) > 2:
        return False
    words = text.split()
    if len(words) < 50:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < CONFIG['min_avg_word_length'] or avg_word_len > CONFIG['max_avg_word_length']:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < CONFIG['min_unique_words_ratio']:
        return False
    lines = text.split('\n')
    if len(lines) > 10:
        unique_lines = len(set(lines))
        if 1 - (unique_lines / len(lines)) > CONFIG['max_line_repetition_ratio']:
            return False
    sentences_end = text.count('.') + text.count('!') + text.count('?')
    if sentences_end < len(words) / 30:
        return False
    return True

# ============================================
# DEDUPLICATOR
# ============================================
class DocumentDeduplicator:
    def __init__(self):
        self.seen_hashes = set()
        self.num_duplicates = 0

    def is_duplicate(self, text: str) -> bool:
        normalized = ' '.join(text.lower().split())
        text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            self.num_duplicates += 1
            return True
        self.seen_hashes.add(text_hash)
        return False

# ============================================
# DOCUMENT TRACKER
# ============================================
class DocumentTracker:
    def __init__(self, initial_pos: int = 0):
        self.boundaries = []
        self.current_pos = initial_pos

    def add_document(self, num_tokens: int):
        self.current_pos += num_tokens
        self.boundaries.append(self.current_pos)

    def find_truncation_point(self, target_tokens: int, tolerance: int) -> int:
        best = 0
        for boundary in self.boundaries:
            if boundary <= target_tokens + tolerance:
                best = boundary
            else:
                break
        return best

# ============================================
# DOWNLOADER
# ============================================
class UltraFilteredDownloader:
    def __init__(self):
        self.output_dir = Path(CONFIG['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print("🔥 Ultra-Filtered Downloader — LLaMA-3 Tokenizer")
        print(f"💾 Checkpoints: tous les {CONFIG['checkpoint_interval']/1e6:.0f}M tokens")
        print(f"📏 Tolérance troncage: ±{CONFIG['token_tolerance']/1e6:.0f}M tokens")
        print(f"⏰ Timeout: {CONFIG['dataset_timeout']/3600:.2f}h par dataset")

        total_per_chunk = sum(d['tokens_per_chunk'] for d in DATASETS)
        print(f"\n📊 Mix par chunk ({total_per_chunk/1e6:.0f}M tokens):")
        for d in DATASETS:
            pct = d['tokens_per_chunk'] / total_per_chunk * 100
            score_info  = f" [int_score>={d['int_score_min']}]" if d['int_score_min'] > 0 else ""
            lang_mode   = d.get('lang_filter_mode', 'none')
            lang_info   = f" [EN:{lang_mode}]"
            filter_info = " [no filter]" if d.get('skip_general_filter') else ""
            print(f"   {d['name']:20s} {d['tokens_per_chunk']/1e6:5.0f}M  ({pct:.0f}%){score_info}{lang_info}{filter_info}")

        print(f"\n🌍 Filtres anglais:")
        print(f"   field    → champ 'language' (FineWeb, FineWiki)")
        print(f"   fasttext → champ 'language_id_whole_page_fasttext' seuil≥{CONFIG['dclm_fasttext_en_threshold']} (DCLM)")
        print(f"   none     → English-only par design officiel (Zyda-2, Cosmopedia, peS2o, FineMath)")

        print(f"\n📝 Loading LLaMA-3 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': SPECIAL_TOKENS
        })
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"   ✅ LLaMA-3 tokenizer ready ({len(self.tokenizer)} tokens)")
        for tok in SPECIAL_TOKENS:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            print(f"      {tok:20s} → {tid}")

        self.state_file = self.output_dir / 'downloader_state.json'
        self.load_state()

        self.offsets_file = Path(CONFIG['offsets_file'])
        self.load_offsets()

    # ── STATE ──────────────────────────────────────────────────────────────
    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            print(f"✅ État chargé: {self.state['completed_chunks']}/{CONFIG['num_chunks']} chunks complétés")
        else:
            self.state = {'completed_chunks': 0}
            print("🆕 Nouvel état créé")

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    # ── OFFSETS ────────────────────────────────────────────────────────────
    def load_offsets(self):
        if self.offsets_file.exists():
            with open(self.offsets_file, 'r') as f:
                self.offsets = json.load(f)
            print(f"✅ Offsets chargés:")
            for name, offset in self.offsets.items():
                print(f"   {name}: {offset:,} docs consommés")
        else:
            self.offsets = {d['name']: 0 for d in DATASETS}
            print("🆕 Offsets initialisés à 0")

    def save_offsets(self):
        with open(self.offsets_file, 'w') as f:
            json.dump(self.offsets, f, indent=2)

    # ── CHECKPOINTS ────────────────────────────────────────────────────────
    def get_existing_checkpoints(self, name: str) -> List[Path]:
        return sorted(self.checkpoint_dir.glob(f"{name}_checkpoint_*.npy"))

    def get_checkpoint_tokens_count(self, checkpoints: List[Path]) -> int:
        total = 0
        for cp in checkpoints:
            arr = np.load(cp, mmap_mode='r')
            total += len(arr)
            print(f"         📂 {cp.name}: {len(arr)/1e6:.1f}M tokens")
        return total

    def save_checkpoint(self, name: str, checkpoint_num: int, tokens: List[int]) -> Path:
        checkpoint_file = self.checkpoint_dir / f"{name}_checkpoint_{checkpoint_num}.npy"
        np.save(checkpoint_file, np.array(tokens, dtype=np.int32))
        return checkpoint_file

    def cleanup_checkpoints(self, name: str):
        checkpoints = self.get_existing_checkpoints(name)
        for cp in checkpoints:
            cp.unlink()
        if checkpoints:
            print(f"      🗑️  {len(checkpoints)} checkpoints supprimés")

    # ── MERGE + TRONCAGE ───────────────────────────────────────────────────
    def merge_and_truncate(
        self,
        checkpoints: List[Path],
        final_tokens: List[int],
        doc_tracker: DocumentTracker,
        target_tokens: int
    ) -> np.ndarray:

        print(f"      🔀 Fusion des checkpoints...")
        all_arrays = []

        for i, cp in enumerate(checkpoints):
            data = np.load(cp)
            all_arrays.append(data)
            print(f"         Checkpoint {i+1}: {len(data)/1e6:.1f}M tokens")

        if final_tokens:
            all_arrays.append(np.array(final_tokens, dtype=np.int32))
            print(f"         RAM finale: {len(final_tokens)/1e6:.1f}M tokens")

        merged = np.concatenate(all_arrays)
        total_before_trunc = len(merged)
        print(f"      📊 Pré-troncage: {total_before_trunc/1e6:.3f}M tokens")

        trunc_idx = doc_tracker.find_truncation_point(target_tokens, CONFIG['token_tolerance'])

        if trunc_idx > 0:
            merged = merged[:trunc_idx]
        else:
            print(f"      ⚠️  Aucune boundary trouvée, troncage simple à {target_tokens/1e6:.0f}M")
            merged = merged[:target_tokens + CONFIG['token_tolerance']]

        deviation = len(merged) - target_tokens
        print(f"      ✂️  Après troncage: {len(merged)/1e6:.3f}M tokens ({deviation:+,} tokens)")

        return merged

    # ── DOWNLOAD DATASET ──────────────────────────────────────────────────
    def download_dataset_for_chunk(self, dataset_config: Dict, chunk_id: int) -> Optional[Dict]:
        name          = dataset_config['name']
        target_tokens = dataset_config['tokens_per_chunk']
        int_score_min = dataset_config.get('int_score_min', 0)
        lang_mode     = dataset_config.get('lang_filter_mode', 'none')
        skip_filter   = dataset_config.get('skip_general_filter', False)

        docs_to_skip = self.offsets.get(name, 0)

        print(f"\n   🎯 {name}: Target {target_tokens/1e6:.0f}M tokens")
        print(f"      📍 Offset: {docs_to_skip:,} docs déjà consommés")
        if int_score_min > 0:
            print(f"      🚀 Fast-reject: int_score < {int_score_min}")
        if lang_mode != 'none':
            print(f"      🇬🇧 Filtre anglais: mode={lang_mode}")
        else:
            print(f"      🇬🇧 Anglais par design (pas de filtre explicite)")
        if skip_filter:
            print(f"      ⚗️  Filtre général désactivé (dataset spécialisé)")

        existing_checkpoints = self.get_existing_checkpoints(name)

        if existing_checkpoints:
            print(f"      🔄 Checkpoints existants trouvés, lecture des tailles réelles...")
        tokens_already_saved = self.get_checkpoint_tokens_count(existing_checkpoints)

        if tokens_already_saved > 0:
            print(f"      🔄 Reprise: {tokens_already_saved/1e6:.1f}M tokens déjà en checkpoints")

        if tokens_already_saved >= target_tokens - CONFIG['token_tolerance']:
            print(f"      ✅ Assez de tokens en checkpoints, fusion directe...")
            doc_tracker = DocumentTracker(initial_pos=0)
            doc_tracker.boundaries.append(tokens_already_saved)
            doc_tracker.current_pos = tokens_already_saved
            return {
                'checkpoints':    existing_checkpoints,
                'final_tokens':   [],
                'doc_tracker':    doc_tracker,
                'target_tokens':  target_tokens,
                'num_docs':       0,
                'num_docs_total': 0,
                'num_docs_read':  0,
                'pass_rate':      0.0,
            }

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(CONFIG['dataset_timeout'])

        try:
            try:
                if dataset_config['config']:
                    dataset = load_dataset(
                        dataset_config['source'],
                        dataset_config['config'],
                        split=dataset_config['split'],
                        streaming=True,
                    )
                else:
                    dataset = load_dataset(
                        dataset_config['source'],
                        split=dataset_config['split'],
                        streaming=True,
                    )
            except Exception as e:
                print(f"   ❌ Erreur chargement {name}: {e}")
                signal.alarm(0)
                return None

            if docs_to_skip > 0:
                print(f"      ⏩ Skip de {docs_to_skip:,} docs...")
                dataset = dataset.skip(docs_to_skip)

            all_tokens   = []
            doc_tracker  = DocumentTracker(initial_pos=tokens_already_saved)
            deduplicator = DocumentDeduplicator() if CONFIG['enable_dedup'] else None

            num_docs_total         = 0
            num_docs_passed        = 0
            num_docs_duplicate     = 0
            num_docs_filtered      = 0
            num_docs_read          = 0
            num_int_score_rejected = 0
            num_language_rejected  = 0

            checkpoint_counter     = len(existing_checkpoints)
            total_tokens_collected = tokens_already_saved
            checkpoint_files       = list(existing_checkpoints)

            text_key = dataset_config['text_key']

            pbar = tqdm(
                total=target_tokens,
                initial=tokens_already_saved,
                desc=f"   {name}",
                unit="tokens",
                unit_scale=True,
            )

            for doc in dataset:
                num_docs_read  += 1
                num_docs_total += 1

                # Fast-reject int_score
                if int_score_min > 0:
                    int_score = doc.get('int_score', -1)
                    if int_score < int_score_min:
                        num_int_score_rejected += 1
                        num_docs_filtered += 1
                        continue

                # Filtre langue (mode adapté par dataset)
                if not is_english(doc, lang_mode):
                    num_language_rejected += 1
                    num_docs_filtered += 1
                    continue

                text = doc.get(text_key, '')
                if not text:
                    continue

                if deduplicator and deduplicator.is_duplicate(text):
                    num_docs_duplicate += 1
                    num_docs_filtered  += 1
                    continue

                if not filter_document(text) and not skip_filter:
                    num_docs_filtered += 1
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)

                doc_tracker.add_document(len(tokens))
                all_tokens.extend(tokens)
                total_tokens_collected += len(tokens)
                num_docs_passed += 1
                pbar.update(len(tokens))

                # Checkpoint logic
                session_tokens           = total_tokens_collected - tokens_already_saved
                expected_new_checkpoints = session_tokens // CONFIG['checkpoint_interval']
                actual_new_checkpoints   = checkpoint_counter - len(existing_checkpoints)

                if expected_new_checkpoints > actual_new_checkpoints:
                    checkpoint_counter += 1
                    cp_path = self.save_checkpoint(name, checkpoint_counter, all_tokens)
                    checkpoint_files.append(cp_path)
                    print(f"\n      💾 Checkpoint {checkpoint_counter}: {len(all_tokens)/1e6:.1f}M → {cp_path.name}")

                    tokens_to_keep = session_tokens % CONFIG['checkpoint_interval']
                    all_tokens = all_tokens[-tokens_to_keep:] if tokens_to_keep > 0 else []

                if total_tokens_collected >= target_tokens + CONFIG['token_tolerance']:
                    break

            pbar.close()
            signal.alarm(0)

            pass_rate    = (num_docs_passed / num_docs_total * 100) if num_docs_total > 0 else 0
            num_rejected = num_docs_total - num_docs_passed

            print(f"   ✅ Collecté: {total_tokens_collected/1e6:.1f}M tokens")
            print(f"      Docs lus: {num_docs_read:,} | Passés: {num_docs_passed:,} ({pass_rate:.1f}%)")
            if num_rejected > 0:
                print(f"      ❌ Rejetés: {num_rejected:,}")
                if num_int_score_rejected > 0:
                    print(f"         - int_score < {int_score_min}: "
                          f"{num_int_score_rejected:,} ({num_int_score_rejected/num_rejected*100:.1f}%) [FAST]")
                if num_language_rejected > 0:
                    print(f"         - Non-English [{lang_mode}]: "
                          f"{num_language_rejected:,} ({num_language_rejected/num_rejected*100:.1f}%)")
                if num_docs_duplicate > 0:
                    print(f"         - Duplicates: {num_docs_duplicate:,} ({num_docs_duplicate/num_rejected*100:.1f}%)")
                remaining = num_docs_filtered - num_int_score_rejected - num_language_rejected
                if remaining > 0:
                    print(f"         - Filtrés:    {remaining:,} ({remaining/num_rejected*100:.1f}%)")

            return {
                'checkpoints':    checkpoint_files,
                'final_tokens':   all_tokens,
                'doc_tracker':    doc_tracker,
                'target_tokens':  target_tokens,
                'num_docs':       num_docs_passed,
                'num_docs_total': num_docs_total,
                'num_docs_read':  num_docs_read,
                'pass_rate':      pass_rate,
            }

        except TimeoutError:
            signal.alarm(0)
            print(f"\n   ⏰ TIMEOUT pour {name}!")
            print(f"      Checkpoints sauvés: {checkpoint_counter} × chunks")
            print(f"      RAM perdue: {len(all_tokens)/1e6:.1f}M tokens")
            print(f"      → Relancez, reprendra au checkpoint {checkpoint_counter}")
            return None
        except Exception as e:
            signal.alarm(0)
            print(f"   ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ── CREATE CHUNK ───────────────────────────────────────────────────────
    def create_chunk(self, chunk_id: int):
        print(f"\n{'='*70}")
        print(f"🔥 CREATING CHUNK {chunk_id + 1}/{CONFIG['num_chunks']}")
        print(f"{'='*70}")

        chunk_dir  = self.output_dir / f"chunk_{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        stats_file = chunk_dir / 'stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                chunk_stats = json.load(f)
            print(f"   📂 Chunk partiellement complété, reprise...")
        else:
            chunk_stats = {
                'chunk_id':      chunk_id,
                'datasets':      {},
                'total_tokens':  0,
                'total_docs':    0,
                'total_size_mb': 0,
                'timestamp':     time.time(),
            }

        for dataset_config in DATASETS:
            name          = dataset_config['name']
            dataset_file  = chunk_dir / f"{name}.npy"
            target_tokens = dataset_config['tokens_per_chunk']

            if dataset_file.exists():
                try:
                    test_load     = np.load(dataset_file, mmap_mode='r')
                    actual_tokens = len(test_load)
                    size_ok       = abs(actual_tokens - target_tokens) <= CONFIG['token_tolerance']
                    if not size_ok:
                        print(f"\n   ⚠️  {name}: Fichier incomplet ({actual_tokens/1e6:.1f}M/{target_tokens/1e6:.0f}M tokens), re-download...")
                        dataset_file.unlink()
                    else:
                        print(f"\n   ✅ {name}: DÉJÀ TÉLÉCHARGÉ ({actual_tokens/1e6:.1f}M tokens) — skip")
                        if name not in chunk_stats['datasets']:
                            chunk_stats['datasets'][name] = {
                                'tokens':  actual_tokens,
                                'size_mb': dataset_file.stat().st_size / (1024**2),
                            }
                            chunk_stats['total_tokens']  += actual_tokens
                            chunk_stats['total_size_mb'] += chunk_stats['datasets'][name]['size_mb']
                        self.cleanup_checkpoints(name)
                        continue
                except Exception:
                    print(f"\n   ⚠️  {name}: Fichier corrompu, re-download...")
                    dataset_file.unlink()

            print(f"\n   🔽 {name}: TÉLÉCHARGEMENT...")
            result = self.download_dataset_for_chunk(dataset_config, chunk_id)

            if result is None:
                print(f"   ⚠️  {name} non complété (timeout/erreur)")
                with open(stats_file, 'w') as f:
                    json.dump(chunk_stats, f, indent=2)
                return None

            merged = self.merge_and_truncate(
                result['checkpoints'],
                result['final_tokens'],
                result['doc_tracker'],
                target_tokens
            )

            np.save(dataset_file, merged)
            size_mb       = dataset_file.stat().st_size / (1024**2)
            actual_tokens = len(merged)

            print(f"      ✅ Sauvegardé: {actual_tokens/1e6:.3f}M tokens ({size_mb:.1f} MB)")

            self.offsets[name] = self.offsets.get(name, 0) + result['num_docs_read']
            self.save_offsets()
            print(f"      📍 Offset mis à jour: {self.offsets[name]:,} docs consommés au total")

            self.cleanup_checkpoints(name)

            chunk_stats['datasets'][name] = {
                'tokens':           actual_tokens,
                'tokens_target':    target_tokens,
                'tokens_deviation': actual_tokens - target_tokens,
                'docs':             result['num_docs'],
                'docs_total':       result['num_docs_total'],
                'pass_rate':        result['pass_rate'],
                'size_mb':          size_mb,
            }
            chunk_stats['total_tokens']  += actual_tokens
            chunk_stats['total_docs']    += result['num_docs']
            chunk_stats['total_size_mb'] += size_mb

            with open(stats_file, 'w') as f:
                json.dump(chunk_stats, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✅ CHUNK {chunk_id + 1} COMPLETED")
        print(f"{'='*70}")
        print(f"📊 Total: {chunk_stats['total_tokens']/1e9:.3f}B tokens | {chunk_stats['total_size_mb']:.1f} MB")

        self.state['completed_chunks'] = chunk_id + 1
        self.save_state()

        return chunk_stats

    # ── RUN ────────────────────────────────────────────────────────────────
    def run(self):
        start_chunk = self.state['completed_chunks']

        print(f"\n🚀 Starting from chunk {start_chunk + 1}/{CONFIG['num_chunks']}")
        print(f"\n📍 Positions actuelles dans les datasets:")
        for name, offset in self.offsets.items():
            print(f"   {name}: {offset:,} docs déjà consommés")

        all_stats = []

        for chunk_id in range(start_chunk, CONFIG['num_chunks']):
            chunk_stats = self.create_chunk(chunk_id)

            if chunk_stats is None:
                print(f"\n⏰ Chunk {chunk_id + 1} interrompu — relancez pour reprendre")
                break

            all_stats.append(chunk_stats)

            import gc
            gc.collect()

        if all_stats:
            print(f"\n{'='*70}")
            print(f"🎉 SESSION TERMINÉE")
            print(f"{'='*70}")
            total_tokens = sum(s['total_tokens'] for s in all_stats)
            total_docs   = sum(s['total_docs']   for s in all_stats)
            total_size   = sum(s['total_size_mb'] for s in all_stats)
            print(f"   Chunks:    {len(all_stats)}")
            print(f"   Tokens:    {total_tokens/1e9:.3f}B")
            print(f"   Documents: {total_docs:,}")
            print(f"   Taille:    {total_size/1024:.2f} GB")
            print(f"\n📂 Données: {self.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-chunks', type=int, default=5)
    args = parser.parse_args()

    CONFIG['num_chunks'] = args.num_chunks

    downloader = UltraFilteredDownloader()
    downloader.run()