# XOver Demo

Independent frontend at https://wanzhuoyue.com/demos/xover/ and pipeline under `xover_demo/`.

The demo runs original C2 `ProfileDppSelector` with the multilingual ml_8x checkpoint, six English examples, alpha=0.3, lambda=0.5, M=20. It executes three candidates (research runs use eight), then calls the original C3 confidence selector without a judge, i.e. nonempty execution voting. C4 uses original output hygiene and one bounded syntax-repair prompt when all candidates have parser/incomplete-output failures. Valid empty results are not repaired. This operating point is displayed in the UI; it does not claim the full research evaluation configuration.

Historical selections and generated DeepSeek candidates can be imported with `scripts/prepare_assets.py`. The importer validates selected indices and exact assembled prompts against the original pool; it never imports test ground-truth queries as generated answers. Candidate provenance remains DeepSeek when reused; new requests use the configured model. Selection, candidate, and complete-answer caches live in `demo/outputs/cache.sqlite3`; map responses reuse the existing bounded map cache. Language, question, bbox, configuration and artifact revisions distinguish cache entries. Restart after replacing research assets.

Deployment imports `xover_demo.api.install(app, settings)` from the existing map API when `XOVER_DEMO_ROOT` is configured. Routes are `/api/xover/config` and `/api/xover/query`. The existing Origin, body size, shared model allowances, execution slots and Turnstile verification apply; one challenge covers the entire bounded query. No model key is shipped to the frontend. The new runtime is `/data/russ/LLM4Geo/XOver_release/demo`; set PYTHONPATH to the existing map backend plus the XOver repository and demo directories when running preparation/tests.

Copy only `frontend/` into the personal site's `demos/xover/`, update its demo-directory card, commit and publish through the site's Jekyll deployment. Do not publish assets/, outputs/ or environment files. The frontend ships local map libraries and fonts with their original licenses.
