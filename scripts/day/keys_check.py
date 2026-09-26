"""QUEUE 2b: were HQ's 'sure' key pairs missed by v2 in v2's candidate set (model rejected) or not (blocking/cascade
cut)? Streams output_v2/candidate_pairs.tsv (light: only the S1s of the sure file are kept in RAM).
Usage: python scripts/day/keys_check.py > runs/day/keys_check.md"""
import pandas as pd

sure = pd.read_csv("scripts/hq_ens/sure_pairs_missed_by_v2.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
need = set(sure["source1_entity_id"])
cand = {}
for ch in pd.read_csv("output_v2/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=200_000):
    ch = ch[ch.iloc[:, 0].isin(need)]
    for s1, cs in zip(ch.iloc[:, 0], ch.iloc[:, 1]):
        cand[s1] = set(cs.split(",")) if cs else set()
sure["in_v2_cands"] = [e in cand.get(s, ()) for s, e in zip(sure["source1_entity_id"], sure["entity_id"])]
sure["status"] = sure["in_v2_cands"].map({True: "model_rejected", False: "blocking_or_cascade_cut"})
print("# Sure key pairs missed by v2: model-rejected vs cut before the model\n")
print(f"{len(sure)} pairs; S1 found in v2 candidate_pairs: {sum(s in cand for s in need)}/{len(need)}\n")
for keys in (["country"], ["country", "rule"], ["country", "rule", "v2_gives_record_to_other_s1"]):
    t = sure.groupby(keys + ["status"]).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    t["cut_share"] = (t.get("blocking_or_cascade_cut", 0) / t["total"]).round(3)
    t = t.sort_values("total", ascending=False)
    print(f"## by {' / '.join(keys)}\n")
    print("```"); print(t.head(40).to_string()); print("```")
    print()
