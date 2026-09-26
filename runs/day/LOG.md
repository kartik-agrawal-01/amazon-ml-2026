# Day loop log

## memory note — 26 Sep (from runs/france_diag)
- Daytime (other users active; mariadbd of another user ~2.3 GB RSS), GPU top-k, --n-jobs 8:
  test/india block 0 was guard-killed (MemAvailable 1270 MB < 1500 MB guard) with **4 stage-B workers**, in a process
  that had already scored France + US (heap not fully returned between countries).
- Rerun of India ALONE in a fresh process with **--stage-b-jobs 3** passed (1242 s for one 100K block).
- Peak RSS: the diag script did not log per-process peak RSS (my omission). Observed system-wide during the US block:
  ~10.1 GB used / 5.5 GB available of 15.6 GB. Reference: v2's own log shows test/india peak RSS 6846 MB (night run, 4 workers).
- => Daytime full-data runs: start with --stage-b-jobs 3 (or 2), and run heavy countries (India) in a fresh process
  when possible; keep the MemAvailable guard at 1.5 GB.
