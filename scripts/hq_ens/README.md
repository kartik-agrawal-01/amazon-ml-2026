# scripts/hq_ens — how HQ built ens2c (26 Sep), prototype scripts (sandbox paths)

Evidence/prototype only: paths are HQ-sandbox paths (`/home/claude/aml`, the staged `submissions/`); the reproducible
pipeline version is `src/hq_keys.py` (QUEUE item). Run everything with `PYTHONHASHSEED=0` (keys use `hash()`).

| step | script | what |
|---|---|---|
| 1 | `keys.py` | normalise ALL train + test records with the new `src/normalize.py`; keep compact join keys per record: `k_core` (core name), `k_nsp` (no-space core), `k_addr` (sorted address tokens), `k_street` (address tokens without numbers), `num1` (first house number), `addr_empty`, legal |
| 2 | `train_calib.py` | full US/India train: all (S1, S2/S3) pairs that share `k_core`, `k_nsp` or `k_addr` (exact-key joins, 12.0M pairs) → category = name relation × address relation × context (`ctx_core_other`: another S1 has the candidate's core name; `ctx_addr_other`: another S1 has the candidate's exact address) → P(match) from the GT |
| 3 | `test_join.py` | same joins/categories on test (11.5M pairs) |
| 4 | `coverage.py`, `vocab_check.py`, `refine_sure.py` | 'sure' categories (train P ≥ 0.96), coverage by v2 / Soha per country; different-name pairs split into real-word vs invented names (tokens outside the country's S1 vocabulary) |
| 5 | `build_ens2c.py` (on top of `build_ens2.py`) | v2 → global one-to-one (keep the owner with a sure key match, else Soha's owner, else none) → add sure pairs whose conservative precision bound (P − coverage)/(1 − coverage) ≥ 0.80 |

Results and tables: `docs/ENSEMBLE_AND_KEYS.md`.
