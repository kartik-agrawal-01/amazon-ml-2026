# Jarvis LOG (memory across sessions)

## Session 1 — 26 Sep 07:30 (machine time)
- Machine: A30 24 GB, 16 vCPU, free reports 503 GB RAM (482 avail), /home 242 GB free. venv /home/venv (torch 2.11 cu130, lightgbm 4.7, pandas 2.3.3).
- Data at data/student_resource/dataset/{train,test} (not data_extracted/; pipeline finds by name under --data-dir data).
- git merge origin/main: already up to date (b68bd46).
- QUEUE 0 (smoke) PLAN: src.pipeline HEAD, all 6 views, k=15, 60K train S1, 60K test S1, block 60000,
  --n-jobs 14 --stage-b-jobs 8 --topk-device cuda, store under /home/cache_jv/store (first run builds the full store = normalisation of 24M rows),
  caches /home/cache_jv/smoke_*. Log -> runs/jarvis/smoke.log. Then project full-run time for 4 vs 6 views, k 10 vs 15.
