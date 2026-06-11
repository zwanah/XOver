#!/usr/bin/env bash
# K-shot sweep: profile-DPP (alpha=0.3, lambda=0.5, M=20) demos, K=1..6, all 8 langs,
# dev split. Generator deepseek-v4-flash NON-THINKING, plain demos, temperature 0.
# K=1..5 cells were sliced from the K=6 greedy selection (prefix-consistent); K=0
# anchor = existing W1 baseline 0-shot. OQS only (no execution).
set -uo pipefail
ROOT=/path/to/XOver
FINAL=$ROOT/final
OUT=$FINAL/outputs/predictions/ovq_kshot
mkdir -p "$OUT/eval_result"
TAG=a0.3_profile_lam0.5_M20

declare -A DIR=(
  [en]=fewshot_profile    [zh]=fewshot_zh_profile  [yue]=fewshot_yue_profile
  [fr]=fewshot_fr_profile [de]=fewshot_de_profile  [ja]=fewshot_ja_profile
  [ko]=fewshot_ko_profile [ru]=fewshot_ru_profile
)
declare -A FULL=(
  [en]=english [zh]=mandarin_simplified [yue]=cantonese [fr]=french
  [de]=german  [ja]=japanese            [ko]=korean    [ru]=russian
)
LANGS=(en zh yue fr de ja ko ru)

for L in "${LANGS[@]}"; do
  for K in 1 2 3 4 5 6; do
    Q="$FINAL/data/${DIR[$L]}/questions_dev_k${K}_${TAG}.json"
    P="$OUT/deepseek-v4-flash_${L}_dev_k${K}_a0.3_lam0.5.json"
    E="$OUT/eval_result/$(basename "${P%.json}")_eval_results.json"
    if [[ ! -f "$Q" ]]; then echo "[$L k$K] MISSING cell $Q"; continue; fi
    if [[ -f "$P" ]]; then
      echo "[$L k$K] pred exists, skip infer"
    else
      echo "---------------- [$L k$K] INFER ----------------"
      ( cd "$FINAL" && python3 scripts/infer_fewshot.py --questions "$Q" \
          --model deepseek-v4-flash --provider deepseek --thinking disabled \
          --mode plain --temperature 0 --workers 128 --num_keys 8 --output "$P" ) \
        || { echo "[$L k$K] INFER FAILED"; continue; }
    fi
    if [[ -f "$E" ]]; then
      echo "[$L k$K] eval exists, skip"
    else
      echo "---------------- [$L k$K] EVAL (OQS) ----------------"
      ( cd "$ROOT" && python3 -m src.eval.runner --input_file "$P" \
          --split dev --lang "${FULL[$L]}" ) || echo "[$L k$K] EVAL FAILED"
    fi
  done
done
echo "################ K-SHOT SWEEP DONE ################"
