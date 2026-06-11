#!/usr/bin/env bash
# Two-phase build of the C3 test candidate pools (K=8), all 8 languages, at the
# ALL-LANGUAGE JOINT OPTIMUM retrieval op-point (alpha=0.3, lambda=0.5) selected
# on dev-OQS via the 6x6 grid (docs/sensitivity-alpha-lambda-results.md).
#
# This MIRRORS run_ml_test_pools.sh (the deployed alpha0.2/lambda0.3 run) but uses
# DISTINCT artifact names so the deployed pools are never overwritten/reused:
#   cell : questions_test_k6_a0.3_profile_lam0.5_M20.json
#   cand : candidates_a0.3_lam0.5_k8.json
#   exec : executed_a0.3_lam0.5_k8.json
#
# PHASE A (C2 generation): 1 greedy(temp0) + 8 samples(temp0.8) per query, all langs.
# PHASE B (C3 execution): execute pools against remote Overpass, english FIRST (gold
#   OverpassQL is shared across langs -> warms the shared gold cache; candidate side
#   reuses the deployed-run cache wherever the generated query is identical).
set -uo pipefail
cd /path/to/XOver

export NO_PROXY=REMOTE_OVERPASS_HOST
export no_proxy=REMOTE_OVERPASS_HOST
export MOSMNL_CLIENT_TIMEOUT=120
REMOTE_OVERPASS='http://REMOTE_OVERPASS_HOST:12347/api/interpreter'
CELL='questions_test_k6_a0.3_profile_lam0.5_M20.json'
TAG='a0.3_lam0.5'
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
  DIR="data/diag_ml_test/$L"; CAND="$DIR/candidates_${TAG}_k8.json"
  QFILE="data/${CELLDIR[$L]}/$CELL"; mkdir -p "$DIR"
  if [[ ! -f "$QFILE" ]]; then echo "[$L] MISSING cell $QFILE, aborting"; exit 1; fi
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
  CAND="$DIR/candidates_${TAG}_k8.json"; EXEC="$DIR/executed_${TAG}_k8.json"
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
