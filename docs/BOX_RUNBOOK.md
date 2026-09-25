# Box runbook — from `git pull` to a validated submission file

Everything below runs on the Ubuntu box in `~/amazon-ml-2026` inside the conda env `aml`.
Start long jobs inside **tmux** (`tmux new -s aml`, detach `Ctrl-b d`, re-attach `tmux a -t aml`).

## 0. One-time setup (5 min)
```bash
cd ~/amazon-ml-2026 && git pull
conda activate aml
bash scripts/gpu_check.sh              # GPU test + installs sparse_dot_topn, pyarrow, joblib
git add -f logs/gpu_box_env.txt && git commit -m "gpu check" && git push
# dataset (zip copied to the box, e.g. into ~/):
mkdir -p data && unzip -o ~/6ab10eb3b23ba_student_resource.zip -d data/data_extracted/
ls data/data_extracted/student_resource/dataset/train      # 4 files expected
```
If `nvidia-smi` still fails, everything below still works on CPU — just drop `--dense`.

## 1. Smoke run (~15–25 min) — proves the code on the box before the big run
```bash
bash scripts/run_pipeline.sh smoke --max-df 0.01 --train-s1 60000 --test-limit 60000 --block-size 60000 --folds 3
```
(Reference: the same settings on Claude's 8%-slice gave OOF macro-F0.5 0.966 / holdout 0.974 — see runs/slice_v1.)
Read `runs/smoke/stdout.txt`: blocking recall table, OOF AUC, `OOF macro F0.5 (chosen)`, breakdown.
Push it: `git add runs/smoke && git commit -m "run smoke" && git push` → HQ reads it.

## 2. Full run v1 (est. 1.5–3 h on 20 threads; GPU makes the dense view cheap)
```bash
bash scripts/run_pipeline.sh v1 --dense --max-df 0.01 --train-s1 300000 --block-size 100000
```
Without a working GPU: `bash scripts/run_pipeline.sh v1 --max-df 0.01 --train-s1 300000 --block-size 100000`.
`--max-df 0.01` matters: with 0.05 the address n-gram products are ~10x bigger (hours, GBs).
Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/report.json`,
`output/model.joblib`; copies of report + results in `runs/v1/`.

## 3. Validate + hand over
```bash
python data/data_extracted/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir data/data_extracted/student_resource/dataset/test
cp output/matching_results.tsv submissions/v1_matching_results.tsv
git add runs/v1 submissions/v1_matching_results.tsv && git commit -m "v1 submission" && git push
```
Then: fill the row in `submissions/LOG.md` (BEFORE upload) → leader pulls → uploads
`submissions/v1_matching_results.tsv` → reports the LB score → fill it in.

## 4. Later layers (only after v1 is on the LB)
```bash
python scripts/finetune_biencoder.py --data-dir data --out models/minilm-er --pairs 300000 --epochs 1
bash scripts/run_pipeline.sh v2_ft --dense --dense-model models/minilm-er --train-s1 300000
```
Each run must beat the previous best `OOF macro F0.5` in `runs/<name>/report.json`, else it is dropped.

## 5. Final package (day 3)
```bash
python scripts/make_package.py --team <team_name> --output output
```

## Memory notes (box has ~10 GB free)
- Caches: `cache/train_norm.pkl`, `cache/test_norm.pkl` (~3–4 GB each on disk, loaded one at a time),
  embeddings `cache/*_emb.npy`. Delete `cache/` to force a rebuild after changing `src/normalize.py`.
- If a run dies with MemoryError: lower `--block-size` to 50000, drop `full_c3` from `--views`
  (`--views name_c3,name_w,addr_c3`), or lower `--train-s1` to 150000.
