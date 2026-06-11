"""Drill into the wrong-but-executable tag errors:
 (a) invalid OSM tag (a tag-validity gate could flag it) vs valid-but-wrong,
 (b) does the K=8 candidate pool already contain a correct candidate? (oracle wedge),
 (c) concrete pred-vs-gold examples per reason class.
"""
import json
import re
import sys
from collections import Counter

BASE = "/path/to/XOver/data/diag_ml_dev"
LANGS = ["en", "zh", "yue", "fr", "de", "ja", "ko", "ru"]
_TAG_RE = re.compile(r'\[\s*"?([\w:]+)"?\s*(!=|!~|=|~)\s*"?([^"\]]+?)"?\s*\]')


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["data"] if isinstance(d, dict) and "data" in d else d


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "1.0")


def kv_eq(q):
    return set((k, v.strip()) for k, op, v in _TAG_RE.findall(q or "") if op == "=")


# Build valid OSM kv vocabulary from the Tag Base used by the TagChecker.
def load_valid_kv():
    try:
        from importlib import import_module
        sys.path.insert(0, "/path/to/XOver")
        from pipeline.c4_revision.tag_kb import TagRetriever
        return set(TagRetriever().valid_kv)
    except Exception as e:
        print(f"(could not load TagRetriever valid_kv: {e})", file=sys.stderr)
        return None


def main():
    valid_kv = load_valid_kv()
    print(f"valid_kv vocab size: {len(valid_kv) if valid_kv else 'N/A'}\n")

    invalid_tag_cases = 0   # pred uses a kv NOT in OSM vocab (validity gate catches)
    valid_wrong_cases = 0   # all pred kv are valid OSM, but != gold kv
    oracle_recoverable = 0  # K=8 pool contains a candidate with correct denotation
    wrong_with_pool = 0
    examples = {"tag_value": [], "area_scope": [], "object_type": []}

    for lang in LANGS:
        pre = load(f"{BASE}/{lang}/eval_result/pred_nonempty_vote_eval_results.json")
        # executed pool (per-candidate EX flags) for oracle-wedge check
        try:
            ep = json.load(open(f"{BASE}/{lang}/executed_lam0.3_k8.json", encoding="utf-8"))
            pool_by_idx = {str(r["query_index"]): r for r in ep["records"]}
        except FileNotFoundError:
            pool_by_idx = {}

        for it in pre:
            if truthy(it["has_error"]) or truthy(it["EX"]):
                continue
            pred, gold = it["generated_text"], it["query_target_ovq"]
            pk, gk = kv_eq(pred), kv_eq(gold)
            if pk != gk and valid_kv is not None:
                bad = [f"{k}={v}" for k, v in pk if f"{k}={v}" not in valid_kv]
                if bad:
                    invalid_tag_cases += 1
                elif pk:
                    valid_wrong_cases += 1
            # oracle wedge: any candidate in the K=8 pool correct?
            pe = pool_by_idx.get(str(it["query_index"]))
            if pe:
                wrong_with_pool += 1
                if any(truthy(c.get("ex")) for c in pe.get("candidates", [])):
                    oracle_recoverable += 1

    print("=== Tag-error decomposition (wrong-but-executable, all langs) ===")
    print(f"  pred uses an INVALID OSM tag (validity gate catches): {invalid_tag_cases}")
    print(f"  pred tags all VALID OSM but wrong for question:        {valid_wrong_cases}")
    print("  -> the second group has NO observable error signal at test time\n")

    if wrong_with_pool:
        print("=== Oracle wedge on the wrong-but-executable population ===")
        print(f"  wrong items with a K=8 pool: {wrong_with_pool}")
        print(f"  pool already contains a CORRECT candidate: {oracle_recoverable} "
              f"({100*oracle_recoverable/wrong_with_pool:.1f}%)")
        print("  -> recoverable by better SELECTION/verification, not by revision prompt\n")


if __name__ == "__main__":
    main()
