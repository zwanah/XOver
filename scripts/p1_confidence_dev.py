"""Dev-EX harness for DeepEye-style confidence selection.

Reuses the existing K-pool artifacts (``executed.json`` + ``candidates.json``)
produced for the voting K-sweep, so dev evaluation needs **no Overpass**: the
picked candidate's EX is read straight from ``executed.json``. The only new cost
is the pairwise LLM judge, which fires solely on *contested* queries (top-1
denotation consistency < ``--threshold``).

It reports EX for the confidence selector alongside the ``nonempty_vote``
baseline on the same pool, plus a route breakdown (shortcut/tournament) and how
often the judge flipped the pick. Judge calls are cached on
``(question, query_a, query_b)`` so re-runs and dev->test stay deterministic and
free.

Usage (from final/):
    python scripts/p1_confidence_dev.py \
        --executed data/diag_p1_en/executed.json \
        --candidates data/diag_p1_en/candidates.json \
        --provider deepseek --model deepseek-v4-flash \
        --k 2 --threshold 0.8 --budget 3 \
        --cache data/diag_p1_en/judge_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.c3_selection import selectors  # noqa: E402
from pipeline.c3_selection.confidence import (  # noqa: E402
    GroupRep, _rank_groups, select_confidence,
)
from pipeline.c3_selection.denotation import freeze  # noqa: E402
from pipeline.c3_selection.evidence import (  # noqa: E402
    CandidateEvidence, contrastive_tags, extract_predicates,
)
from pipeline.c4_revision.llm import OpenAICompatLLM  # noqa: E402


JUDGE_PROMPT = """\
# Task
You are judging two candidate OverpassQL queries written to answer a user's \
natural-language question about OpenStreetMap data. Exactly one is the better \
answer. Decide which query correctly captures what the question asks (the right \
OSM tags / key=value filters, the right area or spatial scope, and the right \
element types).

# Important context
- Candidate A is the higher-confidence option (more of the sampled queries \
agreed on its result). Prefer A unless B is clearly more faithful to the \
question or A has an obvious tagging / scope error.
- The execution result size is a weak hint only (an empty or huge result can \
still be wrong); judge primarily on whether the query's tags and area match \
the question.

# Question
{QUESTION}

# Region / bbox
{REGION}

# Candidate A
Query:
{QUERY_A}
Execution result: {RESULT_A}

# Candidate B
Query:
{QUERY_B}
Execution result: {RESULT_B}

# Output
Respond with XML only:
<result>A</result> or <result>B</result>
"""

def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant pairs (b, c)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


RICH_JUDGE_PROMPT = """\
# Task
Two candidate OverpassQL queries were written to answer a user's \
natural-language question about OpenStreetMap. Exactly one better matches the \
question. Decide based on whether each query's OSM tags (key=value filters), \
area/spatial scope, and element types are faithful to the question — and on \
what each query ACTUALLY MATCHED in the database (the tag frequencies below \
show the real returned elements, which reveals tag/scope mistakes the query \
text alone hides).

# Question
{QUESTION}

# Region
{REGION}

# Candidate A
Predicates: {PRED_A}
Matched: {COUNT_A}; sample matched-element tags: {TAGS_A}

# Candidate B
Predicates: {PRED_B}
Matched: {COUNT_B}; sample matched-element tags: {TAGS_B}

# What differs between what they matched
Only in A: {AONLY}
Only in B: {BONLY}

# Decide
Pick the candidate whose matched elements and predicates are more faithful to \
the question. If genuinely indistinguishable, answer "TIE".
Respond with JSON only, no prose:
{{"choice": "A" | "B" | "TIE"}}
"""

_VOTE_RE = re.compile(r"<result>\s*([AB]|TIE)\s*</result>", re.IGNORECASE | re.DOTALL)
_JSON_CHOICE_RE = re.compile(r'"choice"\s*:\s*"?\s*(A|B|TIE)', re.IGNORECASE)


def parse_vote(resp: str) -> Optional[str]:
    """Extract A/B/TIE from a judge response; tolerate a bare trailing token."""
    if not resp:
        return None
    m = _VOTE_RE.search(resp)
    if m:
        return m.group(1).upper()
    tail = resp.strip().upper()[-6:]
    for tok in ("TIE", "A", "B"):
        if tail.endswith(tok):
            return tok
    return None


def parse_choice_json(resp: str) -> Optional[str]:
    """Extract choice from a JSON-style judge response (rich mode)."""
    if not resp:
        return None
    m = _JSON_CHOICE_RE.search(resp)
    if m:
        return m.group(1).upper()
    return parse_vote(resp)


def card_str(num: int, has_error: bool) -> str:
    if has_error:
        return "execution error"
    n = int(num)
    if n < 0:
        return "execution error"
    if n == 0:
        return "empty result set"
    return f"{n} element(s)"


class JudgeCache:
    """Thread-safe JSON cache of pair-comparison vote lists."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        self._d: Dict[str, List[str]] = {}
        if path and os.path.exists(path):
            try:
                self._d = json.load(open(path))
            except Exception:
                self._d = {}

    @staticmethod
    def key(question: str, qa: str, qb: str) -> str:
        return json.dumps([question, qa, qb], ensure_ascii=False, sort_keys=True)

    def get(self, k: str) -> Optional[List[str]]:
        with self._lock:
            return self._d.get(k)

    def put(self, k: str, votes: List[str]) -> None:
        with self._lock:
            self._d[k] = votes

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            tmp = self.path + ".tmp"
            json.dump(self._d, open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, self.path)


def _evidence_from_cache(rec: Optional[dict], text: str) -> CandidateEvidence:
    """Rebuild a CandidateEvidence from a diag_fetch_evidence cache record."""
    ev = CandidateEvidence(text=text, predicates=extract_predicates(text))
    if rec is None:
        ev.error = "no evidence fetched"
        return ev
    if "error" in rec:
        ev.error = str(rec["error"])[:120]
        return ev
    ev.n_sampled = int(rec.get("n_sampled", 0))
    ev.tag_freq = Counter(rec.get("tag_freq", {}))
    return ev


def make_comparator(llm, client, cache: JudgeCache, *, qi: int, question: str,
                    region: str, texts: List[str], nums: List[int],
                    errs: List[bool], budget: int, judge_input: str,
                    evidence: Optional[Dict[str, dict]] = None):
    """Build a per-query comparator closure over the candidate texts.

    A/B presentation order is swapped on odd ``qi`` so the higher-consistency
    candidate is not always shown as "A" (removes a systematic position bias);
    the swap is deterministic for reproducibility and remapped back to (a, b).
    """
    evidence = evidence or {}
    swap = bool(qi % 2)  # deterministic A/B randomization

    def _result_field(idx: int) -> str:
        if judge_input == "query_only":
            return "(hidden)"
        return card_str(nums[idx], errs[idx])

    def _build_prompt(ia: int, ib: int) -> str:
        if judge_input == "rich":
            ev_a = _evidence_from_cache(evidence.get(f"{qi}:{ia}"), texts[ia])
            ev_b = _evidence_from_cache(evidence.get(f"{qi}:{ib}"), texts[ib])
            a_only, b_only, _ = contrastive_tags(ev_a, ev_b)
            return RICH_JUDGE_PROMPT.format(
                QUESTION=question, REGION=region or "(not specified)",
                PRED_A=ev_a.predicate_str(), COUNT_A=card_str(nums[ia], errs[ia]),
                TAGS_A=ev_a.matched_tags_str(),
                PRED_B=ev_b.predicate_str(), COUNT_B=card_str(nums[ib], errs[ib]),
                TAGS_B=ev_b.matched_tags_str(),
                AONLY=", ".join(a_only) or "(none)",
                BONLY=", ".join(b_only) or "(none)")
        return JUDGE_PROMPT.format(
            QUESTION=question, REGION=region or "(not specified)",
            QUERY_A=texts[ia], RESULT_A=_result_field(ia),
            QUERY_B=texts[ib], RESULT_B=_result_field(ib))

    def comparator(a: GroupRep, b: GroupRep) -> List[str]:
        # Map logical (a=higher consistency, b=lower) to displayed (A, B).
        ia, ib = (b.index, a.index) if swap else (a.index, b.index)
        ck = cache.key(question, texts[ia], texts[ib])
        cached = cache.get(ck)
        if cached is None:
            prompt = _build_prompt(ia, ib)
            msgs = [{"role": "user", "content": prompt}]
            parser = parse_choice_json if judge_input == "rich" else parse_vote
            disp_votes: List[str] = []
            for _ in range(budget):
                v = parser(llm.ask(msgs, client=client))
                if v:
                    disp_votes.append(v)
            cache.put(ck, disp_votes)
        else:
            disp_votes = cached
        if not swap:
            return disp_votes
        # Remap displayed A/B back to logical a/b (a was shown as B).
        return [{"A": "B", "B": "A", "TIE": "TIE"}.get(v, v) for v in disp_votes]

    return comparator


def process_query(rec_ex, rec_txt, llm, client, cache, args, evidence=None):
    cands = [{"sig": freeze(c["sig"]), "ex": int(c["ex"]),
              "num": c.get("num", -1), "has_error": bool(c.get("has_error"))}
             for c in rec_ex["candidates"]]
    texts = rec_txt["candidates"]
    nums = [c.get("num", -1) for c in rec_ex["candidates"]]
    errs = [bool(c.get("has_error")) for c in rec_ex["candidates"]]
    region = rec_txt.get("bbox", "") or ""
    comparator = make_comparator(
        llm, client, cache, qi=rec_ex["query_index"],
        question=rec_txt.get("nl_question", ""),
        region=region, texts=texts, nums=nums, errs=errs,
        budget=args.budget, judge_input=args.judge_input, evidence=evidence)
    r = select_confidence(cands, comparator, k=args.k,
                          shortcut_threshold=args.threshold,
                          prefer_nonempty=not args.no_prefer_nonempty)
    conf_ex = cands[r.picked_index]["ex"] if r.picked_index >= 0 else 0
    ne_ex = selectors.nonempty_vote(cands)
    # Decidable-set leading indicator: among the top-k reps, is exactly one
    # EX-correct? If so the judge faces a well-defined forced choice and we can
    # score its accuracy independent of the EX-invisible (neither-correct) cases.
    reps = _rank_groups(cands, prefer_nonempty=not args.no_prefer_nonempty)
    topk = reps[:args.k]
    n_correct_topk = sum(1 for g in topk if g.ex == 1)
    decidable = int(r.route == "tournament" and n_correct_topk == 1)
    return {"qi": rec_ex["query_index"], "route": r.route,
            "conf_ex": conf_ex, "ne_ex": ne_ex,
            "flipped": int(r.route == "tournament" and conf_ex != ne_ex),
            "decidable": decidable,
            # on a decidable tournament: did the judge land on the correct rep?
            "judge_correct": int(decidable and conf_ex == 1)}


def main() -> None:
    p = argparse.ArgumentParser(description="Confidence-selection dev-EX harness")
    p.add_argument("--executed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--provider", default="deepseek")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--thinking", default="disabled",
                   choices=["auto", "disabled", "enabled"])
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--budget", type=int, default=3, help="judge votes per pair")
    p.add_argument("--judge-input", default="query_card",
                   choices=["query_card", "query_only", "rich"])
    p.add_argument("--evidence", default="",
                   help="evidence cache JSON (required for --judge-input rich)")
    p.add_argument("--no-prefer-nonempty", action="store_true")
    p.add_argument("--num-keys", type=int, default=0)
    p.add_argument("--workers", type=int, default=0,
                   help="concurrent queries (default = #keys)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cache", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default="")
    opts = p.parse_args()

    ex_recs = json.load(open(opts.executed))["records"]
    craw = json.load(open(opts.candidates))
    crecs = craw["records"] if isinstance(craw, dict) and "records" in craw else craw
    txt_by_qi = {r["query_index"]: r for r in crecs}
    if opts.limit:
        ex_recs = ex_recs[:opts.limit]

    evidence = None
    if opts.judge_input == "rich":
        if not opts.evidence or not os.path.exists(opts.evidence):
            p.error("--judge-input rich requires --evidence <cache.json> "
                    "(run scripts/diag_fetch_evidence.py first)")
        evidence = json.load(open(opts.evidence))
        print(f"loaded evidence: {len(evidence)} reps from {opts.evidence}")

    llm = OpenAICompatLLM(model_name=opts.model, provider=opts.provider,
                          temperature=opts.temperature, max_tokens=2048,
                          num_keys=opts.num_keys, thinking=opts.thinking)
    clients = llm.clients
    workers = opts.workers or len(clients)
    cache = JudgeCache(opts.cache or None)

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for i, rec in enumerate(ex_recs):
            client = clients[i % len(clients)]
            futs[pool.submit(process_query, rec, txt_by_qi[rec["query_index"]],
                             llm, client, cache, opts, evidence)] = rec["query_index"]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 50 == 0:
                cache.save()
                print(f"  ... {done}/{len(ex_recs)} queries", flush=True)
    cache.save()

    n = len(results)
    conf = sum(r["conf_ex"] for r in results)
    ne = sum(r["ne_ex"] for r in results)
    routes = {}
    for r in results:
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    n_tourn = routes.get("tournament", 0)
    flips = sum(r["flipped"] for r in results)
    flip_gain = sum(r["conf_ex"] - r["ne_ex"] for r in results if r["route"] == "tournament")
    usage = llm.get_usage()

    # Leading indicator: forced-choice accuracy on the DECIDABLE tournament set
    # (exactly one of the top-k reps is EX-correct). Breakeven = nonempty_vote
    # on the same set (= always pick top-1). good/bad flips + McNemar.
    decidable = [r for r in results if r["decidable"]]
    n_dec = len(decidable)
    judge_correct = sum(r["judge_correct"] for r in decidable)
    ne_correct_dec = sum(r["ne_ex"] for r in decidable)
    good = sum(1 for r in decidable if r["ne_ex"] == 0 and r["conf_ex"] == 1)
    bad = sum(1 for r in decidable if r["ne_ex"] == 1 and r["conf_ex"] == 0)
    # McNemar exact (binomial) on discordant pairs b=bad, c=good.
    mcnemar_p = _mcnemar_exact(bad, good)

    print("\n=== Confidence selection (dev, EX from stored pool) ===")
    print(f"file: {opts.executed}")
    print(f"model={opts.model} provider={opts.provider} thinking={opts.thinking}")
    print(f"k={opts.k} threshold={opts.threshold} budget={opts.budget} "
          f"judge_input={opts.judge_input}")
    print(f"n={n}")
    print(f"nonempty_vote EX : {ne/n:.4f}  ({ne}/{n})")
    print(f"confidence    EX : {conf/n:.4f}  ({conf}/{n})")
    print(f"  delta          : {(conf-ne)/n:+.4f}  ({conf-ne} queries)")
    print(f"routes: {routes}")
    print(f"tournament queries: {n_tourn}  | judge flipped pick: {flips}  "
          f"| net EX change on tournament set: {flip_gain:+d}")
    print("--- leading indicator: DECIDABLE tournament set ---")
    if n_dec:
        print(f"decidable queries : {n_dec}")
        print(f"  judge forced-choice acc : {judge_correct/n_dec:.4f}  "
              f"({judge_correct}/{n_dec})")
        print(f"  breakeven (always-A/NE) : {ne_correct_dec/n_dec:.4f}  "
              f"({ne_correct_dec}/{n_dec})")
        print(f"  good flips (wrong->right): {good}  | "
              f"bad flips (right->wrong): {bad}  | net: {good-bad:+d}")
        print(f"  McNemar exact p         : {mcnemar_p:.4f}")
    else:
        print("  (no decidable tournament queries)")
    print(f"judge token usage: {usage}")

    if opts.out:
        json.dump({"summary": {
            "n": n, "nonempty_vote_ex": ne / n, "confidence_ex": conf / n,
            "delta": (conf - ne) / n, "routes": routes,
            "tournament": n_tourn, "flips": flips, "flip_gain": flip_gain,
            "decidable": {
                "n": n_dec,
                "judge_forced_choice_acc": (judge_correct / n_dec) if n_dec else None,
                "breakeven_acc": (ne_correct_dec / n_dec) if n_dec else None,
                "good_flips": good, "bad_flips": bad, "net": good - bad,
                "mcnemar_exact_p": mcnemar_p},
            "usage": usage,
            "config": {"k": opts.k, "threshold": opts.threshold,
                       "budget": opts.budget, "model": opts.model,
                       "provider": opts.provider, "thinking": opts.thinking,
                       "judge_input": opts.judge_input}},
            "records": results}, open(opts.out, "w"), ensure_ascii=False, indent=2)
        print(f"wrote {opts.out}")


if __name__ == "__main__":
    main()
