"""Thin CLI shim → pipeline.c4_revision.run (C4 conservative revision).

The real implementation lives in the package (``pipeline/c4_revision/run.py``)
and uses package-relative imports, so we invoke it through the package rather
than duplicating it. Equivalent to: ``python -m pipeline.c4_revision.run ...``.

Example:
    python scripts/run_revision.py \
        --input_file outputs/predictions/english_dev_<model>_n100.json \
        --lang english --split dev \
        --model gemini-3-flash-preview --provider aigcbest \
        --temperature 0.7 --max_repair_attempts 3
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.c4_revision.run import main  # noqa: E402

if __name__ == "__main__":
    main()
