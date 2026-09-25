"""Generate a FAKE dataset in the PS format to smoke-test the pipeline end to end.

NOT for modelling decisions — real data will look different. It only exercises the code
paths (I/O, blocking, features, CV, decision rule, writers) and gives runtime estimates.

Layout written (mirrors what we expect from the organisers):
  <out>/train/source1.tsv, source2.tsv, source3.tsv, train_ground_truth.tsv
  <out>/test/source1.tsv,  source2.tsv, source3.tsv
  <out>/test_ground_truth_HIDDEN.tsv   (only so we can score the synthetic test)

Usage: python scripts/make_synthetic.py --out data_synth --n-train 20000 --n-test 8000
"""
import argparse
import os

import numpy as np
import pandas as pd

REGIONS = {
    "US": dict(cities=["San Jose", "Austin", "Reno", "Boise", "Fargo", "Denver", "Miami", "Seattle", "Tulsa", "Omaha"],
               streets=["Market", "Oak", "Pine", "Hill", "Lake", "Elm", "Maple", "Main", "Cedar", "Park"],
               types=[("St", "Street"), ("Ave", "Avenue"), ("Rd", "Road"), ("Blvd", "Boulevard"), ("Dr", "Drive")],
               legal=["Inc", "Inc.", "Incorporated", "LLC", "Corp", "Co", ""], zip_len=5),
    "IN": dict(cities=["Mumbai", "Pune", "Delhi", "Bengaluru", "Jaipur", "Indore", "Nagpur", "Surat"],
               streets=["MG", "Station", "Nehru", "Gandhi", "Tilak", "Ring", "Link", "Hill"],
               types=[("Rd", "Road"), ("Marg", "Marg"), ("Ngr", "Nagar"), ("Ln", "Lane")],
               legal=["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "LLP", "", "Enterprises"], zip_len=6),
    "UK": dict(cities=["London", "Leeds", "Bristol", "Leicester", "York", "Bath"],
               streets=["King", "Queen", "Church", "Mill", "Station", "High"],
               types=[("St", "Street"), ("Rd", "Road"), ("Ln", "Lane")],
               legal=["Ltd", "Limited", "PLC", "LLP", ""], zip_len=0),
    "DE": dict(cities=["Berlin", "Hamburg", "Munich", "Cologne", "Bremen"],
               streets=["Haupt", "Bahnhof", "Garten", "Schul", "Berg"],
               types=[("Str.", "Strasse"), ("Weg", "Weg"), ("Pl.", "Platz")],
               legal=["GmbH", "AG", "KG", "GmbH & Co. KG", ""], zip_len=5),
}
WORDS = ("Acme Delta Bright Zen Kappa Nova Apex Summit Blue Green Silver Golden Royal Prime Metro Urban "
         "Sunrise Lotus Star Omega Alpha Vertex Pioneer Crescent Harbor River Maple Cedar Orion Atlas "
         "Sharma Patel Gupta Mehta Singh Kumar Iyer Rao Khan Das Bose Nair Reddy Joshi Shah").split()
KINDS = ("Robotics Foods Cafe Traders Motors Bakery Textiles Pharma Logistics Systems Electricals Hardware "
         "Studio Clinic Fitness Solutions Jewellers Sweets Printers Builders Opticals Dental Garments").split()
LANDMARKS = ["City Hall", "Railway Station", "Bus Stand", "Central Mall", "Clock Tower", "Main Market", "Temple"]


def typo(s, rng, p=0.3):
    if len(s) < 5 or rng.random() > p:
        return s
    i = rng.integers(1, len(s) - 2)
    op = rng.integers(3)
    if op == 0:
        return s[:i] + s[i + 1:]                       # deletion
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]     # swap
    return s[:i] + "xkq"[rng.integers(3)] + s[i + 1:]  # substitution


def make_entities(n, rng):
    ents = []
    for _ in range(n):
        reg = list(REGIONS)[rng.integers(len(REGIONS))]
        R = REGIONS[reg]
        words = list(rng.choice(WORDS, size=rng.integers(1, 3), replace=False))
        kind = KINDS[rng.integers(len(KINDS))]
        t = R["types"][rng.integers(len(R["types"]))]
        zp = "".join(map(str, rng.integers(0, 10, R["zip_len"]))) if R["zip_len"] else \
            f"{'ABCDLMNS'[rng.integers(8)]}{rng.integers(1, 20)} {rng.integers(1, 9)}{'ABDEF'[rng.integers(5)]}{'GHJLN'[rng.integers(5)]}"
        ents.append(dict(region=reg, words=words, kind=kind, legal=R["legal"][rng.integers(len(R["legal"]))],
                         num=int(rng.integers(1, 999)), street=R["streets"][rng.integers(len(R["streets"]))],
                         stype=t, city=R["cities"][rng.integers(len(R["cities"]))], zip=zp))
    return ents


def render(e, rng, noise):
    """Render one record of entity e. noise: 0 = clean reference, 1 = vendor A, 2 = vendor B (messier)."""
    words = list(e["words"])
    if noise and len(words) > 1 and rng.random() < 0.15:
        words = words[:1]
    name = " ".join(words + [e["kind"]])
    legal = e["legal"]
    if noise and rng.random() < 0.4:
        legal = REGIONS[e["region"]]["legal"][rng.integers(len(REGIONS[e["region"]]["legal"]))]
    if legal:
        name = f"{name} {legal}"
    if noise:
        name = typo(name, rng, 0.25 * noise)
        if rng.random() < 0.2:
            name = name.upper()
    stype = e["stype"][int(rng.random() < 0.5)] if noise else e["stype"][0]
    if noise == 2 and rng.random() < 0.3:          # landmark-style vague address
        addr = f"Nr. {LANDMARKS[rng.integers(len(LANDMARKS))]}, {e['city']}"
    else:
        addr = f"{e['num']} {e['street']} {stype}, {e['city']}"
        if not noise or rng.random() < 0.7:
            addr += f" {e['zip']}"
        if noise:
            addr = typo(addr, rng, 0.15 * noise)
    return name, addr


def build_split(n_s1, rng, prefix_seed):
    n_extra = n_s1 // 2                       # entities that exist only in S2/S3 (distractors)
    ents = make_entities(n_s1 + n_extra, rng)
    # look-alikes: some entities copy another's address (same building) or name stem
    for i in range(0, len(ents), 7):
        j = int(rng.integers(len(ents)))
        if rng.random() < 0.5:
            ents[i].update(num=ents[j]["num"], street=ents[j]["street"], city=ents[j]["city"], zip=ents[j]["zip"],
                           region=ents[j]["region"], stype=ents[j]["stype"])
        else:
            ents[i]["words"] = list(ents[j]["words"])
    ids = rng.choice(np.arange(100000, 1000000), size=4 * len(ents), replace=False)
    it = iter(ids)
    s1, s2, s3, gt = [], [], [], {}
    for k, e in enumerate(ents):
        in_s1 = k < n_s1
        if in_s1:
            sid = f"S1-{next(it)}"
            s1.append((sid, *render(e, rng, 0)))
        matches = []
        n2 = rng.choice([0, 1, 2], p=[0.35, 0.55, 0.10])
        n3 = rng.choice([0, 1, 2], p=[0.45, 0.48, 0.07])
        for _ in range(n2):
            rid = f"S2-{next(it)}"
            s2.append((rid, *render(e, rng, 1)))
            matches.append(rid)
        for _ in range(n3):
            rid = f"S3-{next(it)}"
            s3.append((rid, *render(e, rng, 2)))
            matches.append(rid)
        if in_s1:
            gt[sid] = matches
    cols = ["id", "business_name", "address"]
    shuf = lambda rows: pd.DataFrame(rows, columns=cols).sample(frac=1, random_state=prefix_seed).reset_index(drop=True)
    gt_df = pd.DataFrame({"source1_id": list(gt), "matched_ids": [",".join(v) for v in gt.values()]})
    return shuf(s1), shuf(s2), shuf(s3), gt_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for split, n in [("train", a.n_train), ("test", a.n_test)]:
        s1, s2, s3, gt = build_split(n, rng, a.seed)
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        for i, df in enumerate([s1, s2, s3], 1):
            df.to_csv(os.path.join(d, f"source{i}.tsv"), sep="\t", index=False)
        if split == "train":
            gt.to_csv(os.path.join(d, "train_ground_truth.tsv"), sep="\t", index=False)
        else:
            gt.to_csv(os.path.join(a.out, "test_ground_truth_HIDDEN.tsv"), sep="\t", index=False)
        print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)} singletons={np.mean(gt.matched_ids == ''):.1%}")


if __name__ == "__main__":
    main()
