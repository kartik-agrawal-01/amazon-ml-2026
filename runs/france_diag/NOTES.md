# france_diag — notes (box operator)

Command (tmux, --n-jobs 8, GPU top-k):
  python scripts/france_diag.py score                                  # France + US (India killed by memguard, MemAvailable 1.27 GB)
  python scripts/france_diag.py score --countries india --stage-b-jobs 3   # India alone in a fresh process
  python scripts/france_diag.py report
Runtime: vectoriser refit 214 s; France block 659 s; US 1078 s; India 1242 s; report ~160 s.
Reproduction: block-0 candidate counts, kept pairs and S1-with-matches match runs/v2/stdout.txt exactly;
predicted sets agree 100.0% with output_v2/matching_results.tsv on all 3 x 100K S1 (India 5,905,050 vs 5,905,053 raw
candidates = GPU float ties, no effect after the cascade). Vocab sizes identical to v2.
No src/ file changed. Part E added after soha_matching_results.tsv arrived (python scripts/france_diag.py agree).
See REPORT.md "Summary" for findings.
