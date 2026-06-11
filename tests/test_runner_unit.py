"""
Unit tests for the runner's two MOsmNL-added behaviors:

  1. apply_empty_match_policy — flips EX/EX_soft when both ref and pred
     returned empty sets without erroring. Must NOT flip asymmetric cases.
  2. Overpass.query() hard precondition — raises on non-canonical query.

These tests use no Overpass server, no real cache. They verify the
guard logic in isolation so we don't discover a parse bug mid-batch.
"""
import os
import sys
import shutil
import tempfile
import sqlite3

# Allow `pipeline.c1_eval.runner` import (the consolidated final/ copy).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.c1_eval.runner import apply_empty_match_policy, Overpass  # noqa: E402


def test_empty_match_both_zero_no_error_flips():
    r = {
        'EX': 0,
        'EX_soft': 0.0,
        'info': 'empty_result: ref_num=0, out_num=0',
        'has_error': False,
        'error_type': None,
        'cache_hit': False,
    }
    out = apply_empty_match_policy(r)
    assert out['EX'] == 1, f"expected EX=1, got {out['EX']}"
    assert out['EX_soft'] == 1.0, f"expected EX_soft=1.0, got {out['EX_soft']}"
    assert out['info'] == 'empty_match', f"expected info='empty_match', got {out['info']!r}"
    # has_error/error_type preserved
    assert out['has_error'] is False
    print('OK: both-zero flips to EX=1, EX_soft=1.0, info=empty_match')


def test_empty_match_asymmetric_ref_zero_does_not_flip():
    r = {
        'EX': 0,
        'EX_soft': 0.0,
        'info': 'empty_result: ref_num=0, out_num=5',
        'has_error': False,
        'error_type': None,
        'cache_hit': False,
    }
    out = apply_empty_match_policy(r)
    assert out['EX'] == 0, f"asymmetric ref=0/out=5 must stay EX=0, got {out['EX']}"
    assert out['info'].startswith('empty_result:'), f"info should be untouched, got {out['info']!r}"
    print('OK: ref=0, out=5 stays EX=0')


def test_empty_match_asymmetric_out_zero_does_not_flip():
    r = {
        'EX': 0,
        'EX_soft': 0.0,
        'info': 'empty_result: ref_num=3, out_num=0',
        'has_error': False,
        'error_type': None,
        'cache_hit': False,
    }
    out = apply_empty_match_policy(r)
    assert out['EX'] == 0, f"asymmetric ref=3/out=0 must stay EX=0, got {out['EX']}"
    print('OK: ref=3, out=0 stays EX=0')


def test_empty_match_with_error_does_not_flip():
    r = {
        'EX': 0,
        'EX_soft': 0.0,
        'info': 'pred_exec_error: parse_error',
        'has_error': True,
        'error_type': 'pred_exec_error:parse_error',
        'cache_hit': False,
    }
    out = apply_empty_match_policy(r)
    assert out['EX'] == 0
    assert out['has_error'] is True
    print('OK: has_error=True path preserves EX=0')


def test_empty_match_malformed_info_does_not_flip():
    # Defensive: if upstream changes the info format, we must NOT silently
    # claim empty_match. Returning the original is correct.
    r = {
        'EX': 0,
        'EX_soft': 0.0,
        'info': 'empty_result: SOMETHING WEIRD',
        'has_error': False,
        'error_type': None,
        'cache_hit': False,
    }
    out = apply_empty_match_policy(r)
    assert out['EX'] == 0, "malformed info must leave EX unchanged"
    print('OK: malformed info preserved (no silent EX=1)')


def test_canonical_assert_rejects_nl_prefix():
    """Overpass.query() must raise on non-`[` prefix to avoid cache pollution.

    We construct a real Overpass instance against a temp DB so the
    overridden .query() is exercised without needing a server.
    """
    tmpdir = tempfile.mkdtemp(prefix='mosmnl_test_')
    try:
        # Seed an empty kv table — Overpass __init__ requires the file to exist
        db_path = os.path.join(tmpdir, 'overpass_test_cache.db')
        conn = sqlite3.connect(db_path)
        conn.execute('CREATE TABLE kv(k TEXT PRIMARY KEY, v BLOB NOT NULL, ts REAL NOT NULL);')
        conn.close()

        op = Overpass('http://invalid.local/api', cache_dir=tmpdir,
                      cache_filename='overpass_test_cache',
                      save_frequency=1)

        # Non-canonical: should raise BEFORE touching the network/cache
        try:
            op.query('Here is the Overpass Query:\n\n[out:json];out;')
        except RuntimeError as e:
            assert 'NON-CANONICAL' in str(e), f'unexpected error message: {e}'
            print('OK: NL-prefix query rejected by Overpass.query precondition')
        else:
            raise AssertionError('Overpass.query() should have raised on non-canonical input')

        # Also reject non-string and empty
        for bad in ['', '(node...);', ' [out:json];']:
            try:
                op.query(bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'Should have rejected: {bad!r}')
        print('OK: empty / leading-space / parenthesis queries all rejected')

        op.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    test_empty_match_both_zero_no_error_flips()
    test_empty_match_asymmetric_ref_zero_does_not_flip()
    test_empty_match_asymmetric_out_zero_does_not_flip()
    test_empty_match_with_error_does_not_flip()
    test_empty_match_malformed_info_does_not_flip()
    test_canonical_assert_rejects_nl_prefix()
    print('\nAll runner unit tests passed.')
