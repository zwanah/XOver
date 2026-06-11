"""Tag Base (MTagRet KB) loader + dense retriever for the Stage-4 TagChecker.

A thin, dependency-light tag KB/retriever, kept to
exactly what the tag-repair checker needs: load `data/tag_kb/kb.json`, expose the
valid-tag vocabulary used by the static validity gate, and a cosine retriever over
the precomputed embeddings in `data/tag_index/` for KB-grounded repair candidates.

The encoder is imported lazily so the validity gate (and all unit tests) work
without `sentence_transformers` installed or loaded.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KB_DIR = "/path/to/XOver/data/tag_kb"
DEFAULT_INDEX_DIR = "/path/to/XOver/data/tag_index"
DEFAULT_ENCODER = "paraphrase-multilingual-mpnet-base-v2"


@dataclass(frozen=True)
class TagKBEntry:
    """One retrievable KB entry (mirrors the kb.json schema)."""

    tag_id: str        # "key=value" (equal) or "key=*" (key-level)
    key: str
    value: str         # "" for key-level entries
    kind: str          # "equal" | "key"
    ovq_style: str     # OverpassQL-style quoted form, e.g. '"shop"="florist"'
    has_desc: bool
    text: str          # text fed to the encoder


def load_tag_kb(kb_dir: str = DEFAULT_KB_DIR) -> List[TagKBEntry]:
    """Load the KB entry list from `<kb_dir>/kb.json`."""
    kb_path = os.path.join(kb_dir, "kb.json")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"tag KB missing: {kb_path}")
    with open(kb_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TagKBEntry(**e) for e in raw]


def build_vocab(entries: List[TagKBEntry]) -> Tuple[Set[str], Set[str]]:
    """Derive (valid_keys, valid_kv) from KB entries.

    valid_keys: every key seen in the KB (equal or key-level).
    valid_kv:   the set of `key=value` tag_ids for `kind == "equal"` entries.
    """
    valid_keys = {e.key for e in entries}
    valid_kv = {e.tag_id for e in entries if e.kind == "equal"}
    return valid_keys, valid_kv


def _encoder_slug(encoder_name: str) -> str:
    return encoder_name.replace("/", "__")


def load_tag_index(index_dir: str,
                   encoder_name: str) -> Tuple[np.ndarray, List[str]]:
    """Load (L2-normalized embeddings, tag_ids) for one encoder."""
    slug = _encoder_slug(encoder_name)
    emb_path = os.path.join(index_dir, f"{slug}.npy")
    ids_path = os.path.join(index_dir, f"{slug}_ids.json")
    if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
        raise FileNotFoundError(
            f"tag index missing for encoder={encoder_name}: expected "
            f"{emb_path} and {ids_path}.")
    emb = np.load(emb_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if emb.shape[0] != len(ids):
        raise ValueError(
            f"tag index size mismatch: {emb.shape[0]} embs vs {len(ids)} ids")
    return emb, ids


class TagRetriever:
    """Encode a query, return top-k KB entries by cosine similarity.

    The encoder is loaded lazily; tests inject a fake `_encode`.
    """

    def __init__(self, kb_dir: str = DEFAULT_KB_DIR,
                 index_dir: str = DEFAULT_INDEX_DIR,
                 encoder_name: str = DEFAULT_ENCODER,
                 device: Optional[str] = None) -> None:
        self.kb = load_tag_kb(kb_dir)
        self.valid_keys, self.valid_kv = build_vocab(self.kb)
        self.emb, self.ids = load_tag_index(index_dir, encoder_name)
        by_id = {e.tag_id: e for e in self.kb}
        missing = [i for i in self.ids if i not in by_id]
        if missing:
            raise ValueError(
                f"tag index references {len(missing)} ids absent from KB "
                f"(e.g. {missing[:3]}); rebuild index after KB change")
        self._by_id = by_id
        self._model_name = encoder_name
        self._device = device
        self._encoder = None

    def _encode(self, query: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # heavy import
            self._encoder = SentenceTransformer(self._model_name,
                                                device=self._device)
        vec = self._encoder.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True)
        return vec[0]

    def retrieve(self, query: str, k: int) -> List[Tuple[TagKBEntry, float]]:
        """Top-k KB entries for `query` by cosine similarity."""
        sims = self.emb @ self._encode(query)
        order = np.argsort(-sims)[:k]
        return [(self._by_id[self.ids[int(i)]], float(sims[int(i)]))
                for i in order]
