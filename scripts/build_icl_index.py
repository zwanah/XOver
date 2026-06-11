"""
Build per-language ICL retrieval indices for MOsmNL few-shot LLM baseline.

For each language directory under dataset/OsmNL_<lang>/, concatenates
train_final_<lang>.json + syn_final_<lang>.json (per project memory:
syn is part of the training pool), encodes nl_question with
paraphrase-multilingual-mpnet-base-v2 (L2-normalized), and writes:

    data/icl_index/<lang>.npy        # (N, D) float32, L2-normalized
    data/icl_index/<lang>_keys.json  # list of {sample_index, split, nl_question, OverpassQL}

Mirrors MultiTEND (Qin et al. 2025) App. C: per-language vector library
built from the corresponding-language training set.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LANGS = [
    "english",
    "mandarin_simplified",
    "cantonese",
    "french",
    "german",
    "japanese",
    "korean",
    "russian",
]


def load_train_plus_syn(dataset_root: str, lang: str) -> List[Dict]:
    """Concatenate train_final_<lang>.json + syn_final_<lang>.json."""
    train_path = os.path.join(dataset_root, f"OsmNL_{lang}",
                              f"train_final_{lang}.json")
    syn_path = os.path.join(dataset_root, f"OsmNL_{lang}",
                            f"syn_final_{lang}.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train file missing: {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    syn: List[Dict] = []
    if os.path.exists(syn_path):
        with open(syn_path, "r", encoding="utf-8") as f:
            syn = json.load(f)
    else:
        logger.warning(f"syn file missing for {lang}: {syn_path} (using train only)")
    return train + syn


def build_one(dataset_root: str, out_dir: str, lang: str, model,
              batch_size: int) -> None:
    pool = load_train_plus_syn(dataset_root, lang)
    nls = [it.get("nl_question", "") for it in pool]
    logger.info(f"[{lang}] encoding {len(nls)} NLs (train+syn)")
    embs = model.encode(
        nls,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    if embs.shape[0] != len(pool):
        raise RuntimeError(f"[{lang}] embedding count mismatch: "
                           f"{embs.shape[0]} vs {len(pool)}")

    keys: List[Dict] = []
    for item in pool:
        keys.append({
            "sample_index": item.get("sample_index"),
            "split": item.get("split"),
            "nl_question": item.get("nl_question", ""),
            "OverpassQL": item.get("OverpassQL", ""),
        })

    emb_path = os.path.join(out_dir, f"{lang}.npy")
    keys_path = os.path.join(out_dir, f"{lang}_keys.json")
    np.save(emb_path, embs)
    with open(keys_path, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False)
    logger.info(f"[{lang}] wrote {emb_path} shape={embs.shape} dtype={embs.dtype}")
    logger.info(f"[{lang}] wrote {keys_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",
                   default="/path/to/XOver/dataset")
    p.add_argument("--out_dir",
                   default="/path/to/XOver/data/icl_index")
    p.add_argument("--model",
                   default="paraphrase-multilingual-mpnet-base-v2",
                   help="sBERT model name passed to SentenceTransformer.")
    p.add_argument("--langs", nargs="*", default=None,
                   help=f"Languages to index. Default: {DEFAULT_LANGS}")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None,
                   help="cuda / cpu / cuda:0 ... (auto if None).")
    opts = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")
    os.makedirs(opts.out_dir, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    logger.info(f"loading {opts.model}")
    model = SentenceTransformer(opts.model, device=opts.device)

    langs = opts.langs if opts.langs else DEFAULT_LANGS
    for lang in langs:
        try:
            build_one(opts.dataset_root, opts.out_dir, lang, model, opts.batch_size)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{lang}] failed: {type(e).__name__}: {e}")
            sys.exit(2)


if __name__ == "__main__":
    main()
