"""Stratify each query's candidate pool for the selection diagnostic.

The empty-match policy (any two empty sets count as EX=1) makes a *blended*
Pass@K / voting-accuracy degenerate on gold-empty queries: there, "correct
candidate" == "any candidate returning empty", which most wrong queries do.
So we split queries into strata and gate the headline on S3 only.

* **S0** — gold errored / unusable. Dropped; counted as the usable denominator.
* **S1** — gold empty. Reported separately, EXCLUDED from the headline Pass@K
  (EX is degenerate here).
* **S2** — gold non-empty, all votable candidates share ONE denotation class.
  Voting has nothing to decide → easy.
* **S3** — gold non-empty, >=2 competing denotation classes among votable
  candidates. **This is the experiment**: the voting-vs-oracle gap here is the
  wedge for a learned verifier.

A "denotation class" counts each distinct non-error signature (an empty set is
itself one class — abstaining-by-empty is a vote). Errored candidates are not a
class (they cannot be selected as a denotation).
"""
from __future__ import annotations

from typing import List

from .denotation import Signature, is_error_sig

S0, S1, S2, S3 = "S0", "S1", "S2", "S3"


def n_denotation_classes(candidate_sigs: List[Signature]) -> int:
    """Distinct votable (non-error) denotation signatures in a pool."""
    return len({s for s in candidate_sigs if not is_error_sig(s)})


def classify(
    gold_num: int,
    gold_has_error: bool,
    candidate_sigs: List[Signature],
) -> str:
    """Assign a query to S0/S1/S2/S3.

    Args:
        gold_num: element count of the gold query (-1 ⇒ error, 0 ⇒ empty).
        gold_has_error: True if gold preprocessing/execution failed.
        candidate_sigs: denotation signatures of the K candidates.
    """
    if gold_has_error or int(gold_num) < 0:
        return S0
    if int(gold_num) == 0:
        return S1
    # gold non-empty
    return S3 if n_denotation_classes(candidate_sigs) >= 2 else S2
