"""
MOsmNL eval runner.

Drop-in spiritual replacement for eval_backend/TACO/script/eval_taco_result.py
that:

  1. Uses the unified cache at MOsmNL/data/eval_cache/.
  2. Hard-asserts query.startswith("[") before any op.query(...) call.
  3. Applies the MOsmNL empty-match policy: ref_results == pred_results
     count as EX=1 even if both are empty (cardinality is not a tiebreaker).

CLI is intentionally a strict subset of eval_taco_result.py — only the
flags MOsmNL actually uses. We import the existing Overpass / Nominatim /
OverpassXMLConverter classes from eval_backend so we share the cache
file format. Once we vendor prepare_query (TODO) we can drop the
sys.path hack.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
import tqdm

# Vendor entrypoint: until we copy prepare_query into MOsmNL/src/eval,
# import from eval_backend.
sys.path.insert(0, os.environ.get('EVAL_BACKEND_PATH', '/path/to/eval_backend'))
from utils.evaluation import (  # noqa: E402
    Nominatim,
    OverpassXMLConverter,
    get_exact_match,
    get_oqo_score,
    load_error_cache,
)
from utils.kv_sqlite_cache import Overpass as _BaseOverpass  # noqa: E402
from utils.eval_utils import evaluate_execution_single  # noqa: E402


# Client-side hard cap on a single Overpass HTTP wait, in seconds. Default 500
# preserves the historical behaviour (and the locked baseline EX numbers).
# Set MOSMNL_CLIENT_TIMEOUT=120 (e.g. in the test-pool driver) to cut the
# pathological heavy-query tail: a handful of candidate queries (large area +
# broad match) run for many minutes because the server-side [timeout:300] does
# not abort them at a checkpoint, stalling the worker pool. Queries killed by
# this client cap are tagged with a DISTINCT 'client_timeout' marker so the
# count of force-killed queries stays recoverable from the candidate pool
# (vs the base class's 'server_error_timeout'); both are num=-1 errors
# downstream, so EX/voting treat them identically.
CLIENT_TIMEOUT = int(os.environ.get('MOSMNL_CLIENT_TIMEOUT', '500'))


class Overpass(_BaseOverpass):
    """Overpass wrapper that enforces the canonical-key precondition
    (plan §6 R1). Any query reaching .query() that does not start with
    '[' is a sign that prepare_query was bypassed; we refuse to execute
    rather than pollute the cache with NL-prefix garbage.

    Also enforces a client-side wait cap (``CLIENT_TIMEOUT``) and tags
    client-killed queries with a distinct ``client_timeout`` marker.
    """
    def query(self, query, use_cache=True, timeout=None):  # noqa: A003 (shadow OK)
        if not isinstance(query, str) or not query.startswith('['):
            raise RuntimeError(
                f'NON-CANONICAL query reached Overpass.query(); refused. '
                f'head={query[:120]!r}'
            )
        if timeout is None:
            timeout = CLIENT_TIMEOUT
        if use_cache and query in self.cache:
            self.num_cache_hits += 1
            cached = self.cache.get(query)
            return tuple(cached) if isinstance(cached, (list, tuple)) else cached
        try:
            result = self._query(query, timeout)
        except requests.ConnectionError:
            raise
        except requests.Timeout:
            result = ('client_timeout', -1)  # our client cap fired (distinct marker)
        except Exception:
            result = ('unknown_error', -2)
        self.add_cache(query, result)
        return result

logger = logging.getLogger(__name__)


# ---- MOsmNL fixed paths -----------------------------------------------------
MOSMNL_ROOT = '/path/to/XOver'
DEFAULT_CACHE_DIR = os.path.join(MOSMNL_ROOT, 'data/eval_cache')
DEFAULT_OVERPASS_URLS = (
    'http://localhost:12346/api/interpreter,'
    'http://REMOTE_OVERPASS_HOST:12347/api/interpreter'
)
DEFAULT_NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search.php'
DEFAULT_CONVERT_URL = 'http://localhost:12346/api/convert'
TIMEOUT_SEC = 300


# ---- bbox load -------------------------------------------------------------
def load_bbox_from_mosmnl(split: str, lang: str) -> dict:
    """Read bbox map from the MOsmNL multilingual dataset.

    MOsmNL/dataset/OsmNL_<lang>/<split>_final_<lang>.json carries the same
    location info as the original OsmNL/Final/<split>_final.json (the
    bbox is invariant across translations).
    """
    path = os.path.join(
        MOSMNL_ROOT, 'dataset', f'OsmNL_{lang}',
        f'{split}_final_{lang}.json',
    )
    if not os.path.exists(path):
        # Fallback: original english dataset.
        path = f'/path/to/OsmNL_source/{split}_final.json'
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    out = {}
    for it in items:
        idx = it.get('sample_index', -1)
        loc = it.get('location', {})
        bbox = loc.get('bbox', '')
        if bbox in ('none', None):
            bbox = ''
        out[idx] = bbox
    return out


# ---- precondition guard ----------------------------------------------------
def assert_canonical(prepared_query: str, label: str, sample_idx: int):
    """R1: a query that reaches op.query(...) MUST be canonicalized
    (starts with '['). Anything else means prepare_query was bypassed
    and we are about to pollute the cache.
    """
    if not prepared_query.startswith('['):
        raise RuntimeError(
            f'NON-CANONICAL query reached op.query(); refuse to execute. '
            f'label={label} sample_idx={sample_idx} head={prepared_query[:120]!r}'
        )


# ---- empty-match policy ----------------------------------------------------
def apply_empty_match_policy(ex_result: Dict) -> Dict:
    """MOsmNL plan §3a:
    if both ref and pred returned 0 elements and neither errored,
    treat as EX=1, EX_soft=1.0 (empty set matches empty set).

    Detects the 'empty_result' branch from evaluate_execution_single,
    where info == 'empty_result: ref_num=0, out_num=0' and has_error
    is False.
    """
    if ex_result.get('has_error'):
        return ex_result
    info = ex_result.get('info', '')
    if info.startswith('empty_result:'):
        # Parse ref_num and out_num from the info string.
        # Format: 'empty_result: ref_num=N, out_num=M'
        try:
            ref_part, out_part = info.replace('empty_result:', '').split(',')
            ref_num = int(ref_part.strip().split('=')[1])
            out_num = int(out_part.strip().split('=')[1])
        except Exception:
            return ex_result
        if ref_num == 0 and out_num == 0:
            new = dict(ex_result)
            new['EX'] = 1
            new['EX_soft'] = 1.0
            new['info'] = 'empty_match'
            return new
    return ex_result


# ---- main driver -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='MOsmNL eval runner')
    parser.add_argument('--input_file', required=True,
                        help='Path to a results JSON (TACO-style format).')
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--split', default=None, choices=['dev', 'test'],
                        help='If omitted, inferred from input filename.')
    parser.add_argument('--lang', default='english',
                        help='Language token for bbox lookup '
                             '(e.g. english, mandarin_simplified, cantonese, '
                             'french, german, japanese, korean, russian).')
    parser.add_argument('--compute_execution', action='store_true')
    parser.add_argument('--retry_errors', default=False,
                        type=lambda x: str(x).lower() in ('true', 'yes', '1'))
    parser.add_argument('--cache_dir', default=DEFAULT_CACHE_DIR)
    parser.add_argument('--overpass_urls', default=DEFAULT_OVERPASS_URLS)
    parser.add_argument('--nominatim_url', default=DEFAULT_NOMINATIM_URL)
    parser.add_argument('--convert_url', default=DEFAULT_CONVERT_URL)
    parser.add_argument('--cache_save_frequency', type=int, default=100)
    opts = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(message)s',
    )

    # Resolve split
    if opts.split is None:
        fn = os.path.basename(opts.input_file).lower()
        if 'dev' in fn:
            opts.split = 'dev'
        elif 'test' in fn:
            opts.split = 'test'
        else:
            raise SystemExit('Cannot infer --split from filename; pass explicitly.')
    split = opts.split

    # Resolve output path
    if opts.output_file is None:
        in_dir = os.path.dirname(os.path.abspath(opts.input_file))
        in_name = os.path.splitext(os.path.basename(opts.input_file))[0]
        eval_dir = os.path.join(in_dir, 'eval_result')
        os.makedirs(eval_dir, exist_ok=True)
        opts.output_file = os.path.join(eval_dir, f'{in_name}_eval_results.json')

    # Endpoints
    overpass_urls = [u.strip() for u in opts.overpass_urls.split(',') if u.strip()]
    logger.info(f'Overpass endpoints ({len(overpass_urls)}): {overpass_urls}')
    overpass_instances = [
        Overpass(url, cache_dir=opts.cache_dir,
                 cache_filename=f'overpass_{split}_cache',
                 save_frequency=opts.cache_save_frequency)
        for url in overpass_urls
    ]
    nominatim = Nominatim(opts.nominatim_url, cache_dir=opts.cache_dir,
                          save_frequency=opts.cache_save_frequency)
    converter = OverpassXMLConverter(opts.convert_url,
                                     cache_dir=opts.cache_dir,
                                     save_frequency=opts.cache_save_frequency)

    # Load input
    with open(opts.input_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
    if isinstance(file_data, dict) and 'data' in file_data:
        data = file_data['data']
        in_stats = file_data.get('statistics')
    elif isinstance(file_data, list):
        data = file_data
        in_stats = None
    else:
        raise SystemExit(f'unknown input format: {type(file_data)}')
    total = len(data)
    if total == 0:
        raise SystemExit('input file has 0 samples')
    # Schema contract: every result file MUST carry these field names.
    required_fields = {'query_index', 'query_target_ovq', 'generated_text'}
    missing = required_fields - set(data[0].keys())
    if missing:
        raise SystemExit(
            f'input sample is missing required fields {sorted(missing)}; '
            f'got keys {sorted(data[0].keys())}. '
            f'Fix the LLM prediction-export script to emit '
            f'query_index / query_target_ovq / generated_text.'
        )
    logger.info(f'samples: {total}; split={split}; lang={opts.lang}')

    bbox_map = load_bbox_from_mosmnl(split, opts.lang)

    # Step 1: batch OQS
    all_pred = [s.get('generated_text', '') for s in data]
    all_ref = [s.get('query_target_ovq', '') for s in data]
    try:
        oqo_pack = get_oqo_score(all_pred, all_ref, converter, weights=[1, 1, 1])
        oqs_scores = oqo_pack[1]
    except Exception as e:
        logger.error(f'OQS batch failed: {e}')
        oqs_scores = {'oqo': [0.0]*total, 'kv_overlap': [0.0]*total,
                      'xml_overlap': [0.0]*total, 'chrf': [0.0]*total}

    # Step 2: EM
    em_score, em_list = get_exact_match(all_pred, all_ref)
    exact = sum(em_list)
    results = []
    for idx, sample in enumerate(tqdm.tqdm(data, desc='assemble')):
        qidx = sample.get('query_index', -1)
        bbox = bbox_map.get(qidx, sample.get('bbox', ''))
        results.append({
            'query_index': qidx,
            'query_nl': sample.get('query_nl', ''),
            'query_target_ovq': sample.get('query_target_ovq', ''),
            'generated_text': sample.get('generated_text', ''),
            'bbox': bbox,
            'EM': em_list[idx],
            'OQS': round(oqs_scores['oqo'][idx], 4),
            'kv_overlap': round(oqs_scores['kv_overlap'][idx], 4),
            'xml_overlap': round(oqs_scores['xml_overlap'][idx], 4),
            'chrf': round(oqs_scores['chrf'][idx], 4),
        })

    # Step 3: optional execution
    if opts.compute_execution:
        logger.info('Running execution (EX / EX_soft)...')
        error_cache = load_error_cache(opts.cache_dir)
        pending = [i for i, r in enumerate(results) if not r['EM']]
        for i, r in enumerate(results):
            if r['EM']:
                r['EX'] = 1
                r['EX_soft'] = 1.0
                r['execution_info'] = 'skipped_em_match'
                r['has_error'] = False
        logger.info(f'skipped EM=1: {total-len(pending)}; need to execute: {len(pending)}')

        def process(idx):
            r = results[idx]
            ex = evaluate_execution_single(
                r['query_target_ovq'], r['generated_text'], r['bbox'],
                False, error_cache, idx,
                TIMEOUT_SEC, opts.retry_errors,
                overpass_instances, nominatim,
                include_error_tracking=True,
            )
            ex = apply_empty_match_policy(ex)
            return idx, ex

        with ThreadPoolExecutor(max_workers=len(overpass_instances)) as ex_pool:
            futures = {ex_pool.submit(process, i): i for i in pending}
            for fut in tqdm.tqdm(as_completed(futures), total=len(pending), desc='execute'):
                idx, ex = fut.result()
                results[idx]['EX'] = ex['EX']
                results[idx]['EX_soft'] = ex['EX_soft']
                results[idx]['execution_info'] = ex.get('info', '')
                results[idx]['has_error'] = ex.get('has_error', False)
                results[idx]['error_type'] = ex.get('error_type', None)

        # checkpoint caches
        for op in overpass_instances:
            op.save_cache()
            op.close()
        nominatim.save_cache()
    else:
        for r in results:
            r['EX'] = None
            r['EX_soft'] = None
            r['execution_info'] = 'execution_not_computed'
            r['has_error'] = None

    converter.save_cache()

    # Compose statistics
    em_pct = round(exact / total * 100, 2) if total else 0.0
    avg_oqs = sum(r['OQS'] for r in results) / total if total else 0.0
    eval_stats = {
        'EM': {'count': exact, 'total': total, 'rate': em_pct},
        'OQS': {'mean': round(avg_oqs, 4)},
    }
    if opts.compute_execution:
        ex_vals = [r['EX'] for r in results if r['EX'] is not None]
        if ex_vals:
            eval_stats['EX'] = {
                'count': sum(ex_vals),
                'total': len(ex_vals),
                'rate': round(sum(ex_vals) / len(ex_vals), 4),
            }
        soft_vals = [r['EX_soft'] for r in results if r['EX_soft'] is not None]
        if soft_vals:
            eval_stats['EX_soft'] = {'mean': round(sum(soft_vals) / len(soft_vals), 4)}

    out_data: Dict[str, Any] = {'data': results}
    if in_stats is not None:
        out_data['statistics'] = copy.deepcopy(in_stats)
        out_data['statistics']['evaluation'] = eval_stats
    else:
        out_data['statistics'] = {
            'total_queries': total,
            'split': split,
            'lang': opts.lang,
            'evaluation': eval_stats,
        }

    with open(opts.output_file, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info(f'EM: {exact}/{total} = {em_pct}%')
    logger.info(f'OQS mean: {avg_oqs:.4f}')
    if opts.compute_execution and 'EX' in eval_stats:
        logger.info(f"EX: {eval_stats['EX']['count']}/{eval_stats['EX']['total']} = "
                    f"{eval_stats['EX']['rate']:.4f}")
        logger.info(f"EX_soft mean: {eval_stats['EX_soft']['mean']:.4f}")
    logger.info(f'wrote: {opts.output_file}')


if __name__ == '__main__':
    main()
