#!/usr/bin/env bash
# Two-phase build of the C3 test candidate pools (K=8) for all 8 languages.
#
#   PHASE A (C2 generation): generate 1 greedy + 8 temp-0.8 samples per query
#     for every language up front (API-bound, ~4 min/lang). Decoupled from
#     execution so generation never waits behind the slow Overpass pass.
#   PHASE B (C3 execution): execute every pool against Overpass, english FIRST
#     (gold OverpassQL is identical across languages, so english warms the
#     shared overpass_test_cache gold side for langs 2..8).
#
# Execution uses remote-ONLY (:12347) at 16 workers with a 120s CLIENT cap:
# the local docker OOMs under concurrency, and a few candidate queries run for
# many minutes (server-side [timeout:300] doesn't abort them), stalling the
# pool. Queries killed by the client cap are tagged 'client_timeout' and counted.
set -uo pipefail
cd /path/to/XOver

export NO_PROXY=REMOTE_OVERPASS_HOST
export no_proxy=REMOTE_OVERPASS_HOST
export MOSMNL_CLIENT_TIMEOUT=120
REMOTE_OVERPASS='http://REMOTE_OVERPASS_HOST:12347/api/interpreter'
CELL='questions_test_k6_a0.2_profile_lam0.3_M20.json'
GEN_WORKERS=128
NUM_KEYS=8
EXEC_WORKERS=16

declare -A CELLDIR=(
  [en]=fewshot_profile      [zh]=fewshot_zh_profile   [yue]=fewshot_yue_profile
  [fr]=fewshot_fr_profile   [de]=fewshot_de_profile   [ja]=fewshot_ja_profile
  [ko]=fewshot_ko_profile   [ru]=fewshot_ru_profile
)
LANGS=(en zh yue fr de ja ko ru)   # english first (matters for PHASE B)

# ---------------- PHASE A: generation ----------------
echo "################ PHASE A: C2 GENERATION (all langs) ################"
for L in "${LANGS[@]}"; do
  DIR="data/diag_ml_test/$L"; CAND="$DIR/candidates_lam0.3_k8.json"
  QFILE="data/${CELLDIR[$L]}/$CELL"; mkdir -p "$DIR"
  if [[ -f "$CAND" ]]; then echo "[$L] candidates exist, skip gen"; continue; fi
  echo "---------------- [$L] GENERATE ----------------"
  python3 scripts/diag_gen_candidates.py \
    --questions "$QFILE" --n_queries 1000 --k 8 --temperature 0.8 --seed 42 \
    --workers "$GEN_WORKERS" --num_keys "$NUM_KEYS" --resume --output "$CAND"
  if [[ $? -ne 0 || ! -f "$CAND" ]]; then echo "[$L] GEN FAILED, aborting"; exit 1; fi
done
echo "################ PHASE A DONE: all candidates generated ################"

# ---------------- PHASE B: execution ----------------
echo "################ PHASE B: C3 EXECUTION (english first) ################"
for L in "${LANGS[@]}"; do
  DIR="data/diag_ml_test/$L"
  CAND="$DIR/candidates_lam0.3_k8.json"; EXEC="$DIR/executed_lam0.3_k8.json"
  if [[ -f "$EXEC" ]]; then echo "[$L] executed exists, skip"; continue; fi
  echo "---------------- [$L] EXECUTE ----------------"
  python3 scripts/diag_execute.py \
    --input "$CAND" --output "$EXEC" \
    --split test --overpass_urls "$REMOTE_OVERPASS" --workers "$EXEC_WORKERS"
  if [[ $? -ne 0 || ! -f "$EXEC" ]]; then echo "[$L] EXEC FAILED, aborting"; exit 1; fi
  python3 - "$EXEC" "$L" <<'PY'
import json, sys
exf, lang = sys.argv[1], sys.argv[2]
d = json.load(open(exf)); recs = d["records"]
cand_ct = sum(1 for r in recs for c in r["candidates"]
              if c["sig"][:2] == ["error", "client_timeout"])
gold_ct = sum(1 for r in recs if r["gold_sig"][:2] == ["error", "client_timeout"])
n_cand = sum(len(r["candidates"]) for r in recs)
all_killed = sum(1 for r in recs if r["candidates"] and
                 all(c["sig"][:2] == ["error", "client_timeout"] for c in r["candidates"]))
print(f"[{lang}] CLIENT_TIMEOUT count: candidates {cand_ct}/{n_cand} "
      f"({100*cand_ct/max(1,n_cand):.2f}%), gold {gold_ct}/{len(recs)}, "
      f"queries-all-killed {all_killed}/{len(recs)}")
PY
  echo "[$L] DONE -> $EXEC"
done

echo "ALL 8 LANGUAGES DONE"
