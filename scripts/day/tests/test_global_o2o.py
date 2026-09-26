"""Light unit test of pipeline.decided_pairs + global_o2o across two blocks. python scripts/day/tests/test_global_o2o.py"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.pipeline import decided_pairs, global_o2o  # noqa: E402

d_rid = np.array(["S2-a", "S2-b", "S3-c"], dtype=object)
# block 1: S1-1 keeps a (0.9) and c (0.8); block 2: S1-2 keeps a (0.95) and b (0.7); S1-3 keeps nothing
b1 = pd.DataFrame({"q": [0, 0, 0], "c": [0, 1, 2], "p": [0.9, 0.3, 0.8]})
b2 = pd.DataFrame({"q": [0, 0, 1], "c": [0, 1, 2], "p": [0.95, 0.7, 0.4]})
q1, q2 = np.array(["S1-1"], dtype=object), np.array(["S1-2", "S1-3"], dtype=object)
s1 = {"S1-1": frozenset({"S2-a", "S3-c"})}
s2 = {"S1-2": frozenset({"S2-a", "S2-b"}), "S1-3": frozenset()}
dp = [decided_pairs(s1, b1, q1, d_rid), decided_pairs(s2, b2, q2, d_rid)]
sets = {**s1, **s2}
n = global_o2o(sets, *(np.concatenate([d[i] for d in dp]) for i in range(3)))
assert n == 1, n
assert sets == {"S1-1": frozenset({"S3-c"}), "S1-2": frozenset({"S2-a", "S2-b"}), "S1-3": frozenset()}, sets
print("global_o2o OK", sets)
