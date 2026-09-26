import pandas as pd, numpy as np, os, pickle, time
SUB = "/mnt/user-data/uploads/Amazon ML Comp/submissions"
def read_res(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            s1, _, m = line.rstrip("\n").rstrip("\r").partition("\t")
            d[s1] = tuple(x for x in m.split(",") if x) if m else ()
    return header, d
