"""Analyze C4 revision: where does generation go wrong, and can the reviser see it?

For each language: load pre-revision eval (nonempty_vote) and post-revision eval.
Categorize the 1000 queries by (exec_ok, EX) and dissect the failure populations
against gold (query_target_ovq).
"""
import json
import re
import sys
from collections import Counter

BASE = "/path/to/XOver/data/diag_ml_dev"
LANGS = ["en", "zh", "yue", "fr", "de", "ja", "ko", "ru"]

_TAG_RE = re.compile(r'\[\s*"?([\w:]+)"?\s*(!=|!~|=|~)\s*"?([^"\]]+?)"?\s*\]')
_AREA_RE = re.compile(r'geocodeArea:"([^"]+)"|nominatimArea:"([^"]+)"|area\(([0-9]+)\)')
_OBJ_RE = re.compile(r'\b(nwr|node|way|rel|relation)\b')


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["data"] if isinstance(d, dict) and "data" in d else d


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "1.0")


def tags_of(q):
    return set((k, op, v.strip()) for k, op, v in _TAG_RE.findall(q or ""))


def keys_of(q):
    return set(k for k, op, v in _TAG_RE.findall(q or ""))


def kv_of(q):
    return set((k, v.strip()) for k, op, v in _TAG_RE.findall(q or "") if op == "=")


def areas_of(q):
    out = set()
    for a, b, c in _AREA_RE.findall(q or ""):
        out.add(a or b or ("area#" + c))
    return out


def objs_of(q):
    return Counter(m for m in _OBJ_RE.findall(q or ""))


def classify_wrong(pred, gold):
    """Why does an executable pred get EX=0? Diff structural facets vs gold."""
    reasons = []
    pk, gk = kv_of(pred), kv_of(gold)
    if pk != gk:
        missing = gk - pk
        extra = pk - gk
        if missing or extra:
            reasons.append("tag_value")
    pa, ga = areas_of(pred), areas_of(gold)
    if pa != ga:
        reasons.append("area_scope")
    po, go = objs_of(pred), objs_of(gold)
    if set(po) != set(go):
        reasons.append("object_type")
    # filter operator mismatch (= vs ~ vs !=) on shared keys
    pops = set((k, op) for k, op, v in _TAG_RE.findall(pred or ""))
    gops = set((k, op) for k, op, v in _TAG_RE.findall(gold or ""))
    if pops != gops and not reasons:
        reasons.append("filter_op")
    # around/distance
    if ("around" in (gold or "")) != ("around" in (pred or "")):
        reasons.append("spatial_around")
    if not reasons:
        reasons.append("other/structural")
    return reasons


def main():
    print(f"{'lang':5} {'n':>4} {'EX=1':>5} {'execERR':>7} {'exec_ok&EX0':>11} "
          f"{'%wrong-but-exec':>15}")
    print("-" * 60)
    grand_wrong = Counter()
    grand_revised = []
    for lang in LANGS:
        pre = load(f"{BASE}/{lang}/eval_result/pred_nonempty_vote_eval_results.json")
        n = len(pre)
        ex1 = sum(1 for it in pre if truthy(it["EX"]))
        exec_err = sum(1 for it in pre if truthy(it["has_error"]))
        # executable but wrong: not error, EX==0
        wrong_exec = [it for it in pre
                      if not truthy(it["has_error"]) and not truthy(it["EX"])]
        print(f"{lang:5} {n:>4} {ex1:>5} {exec_err:>7} {len(wrong_exec):>11} "
              f"{100*len(wrong_exec)/n:>14.1f}%")
        for it in wrong_exec:
            for r in classify_wrong(it["generated_text"], it["query_target_ovq"]):
                grand_wrong[r] += 1

        # revised population: post file with revision metadata
        try:
            post = load(f"{BASE}/{lang}/revised/pred_nonempty_vote.json")
            posteval = load(f"{BASE}/{lang}/revised/eval_result/pred_nonempty_vote_eval_results.json")
            eval_by_idx = {str(it["query_index"]): it for it in posteval}
            for it in post:
                rev = it.get("revision", {})
                if rev.get("repaired"):
                    pe = eval_by_idx.get(str(it["query_index"]), {})
                    grand_revised.append({
                        "lang": lang,
                        "qidx": it["query_index"],
                        "ex_after": truthy(pe.get("EX", 0)),
                    })
        except FileNotFoundError:
            pass

    print("\n=== WHY executable predictions are WRONG (EX=0, all langs pooled) ===")
    total = sum(grand_wrong.values())
    for r, c in grand_wrong.most_common():
        print(f"  {r:20} {c:>5}  ({100*c/total:.1f}% of reason-hits)")

    print(f"\n=== REVISED cases (C4 actually touched), all langs ===")
    print(f"  total repaired: {len(grand_revised)}")
    became = sum(1 for r in grand_revised if r["ex_after"])
    print(f"  became correct after repair: {became}/{len(grand_revised)}")


if __name__ == "__main__":
    main()
