import pickle, sys, collections
import numpy as np, pandas as pd

def load_keys(split):
    parts = [pd.read_pickle(f"/home/claude/aml/ens/keys_{split}_s{s}.pkl") for s in (1, 2, 3)]
    return parts

def tok_rel(a, b):
    A, B = a.split(), b.split(); SA, SB = set(A), set(B)
    if not SA or not SB: return "n_empty"
    if SA == SB: return "n_reorder"
    j = len(SA & SB) / len(SA | SB)
    if SA < SB or SB < SA: return "n_subset"
    if len(SA) == len(SB) and len(SA - SB) == 1: return "n_swap1"
    if j == 0: return "n_disjoint"
    return "n_partial"

def pair_cats(q, c, owners_addr, owners_street_core):
    """q, c: dicts (rows) with keys. Returns (name_cat, addr_cat, ctx)."""
    if q["k_core"] == c["k_core"]: nc = "n_core_eq"
    elif q["k_nsp"] == c["k_nsp"]: nc = "n_nsp_eq"
    else: nc = tok_rel(q["n_core"], c["n_core"])
    if c["addr_empty"]: ac = "a_c_empty"
    elif q["k_addr"] == c["k_addr"]: ac = "a_eq"
    elif q["k_street"] == c["k_street"]:
        ac = "a_street_eq_num_diff" if (q["num1"] >= 0 and c["num1"] >= 0) else "a_street_eq_num_missing"
    elif q["num1"] >= 0 and q["num1"] == c["num1"]: ac = "a_num_eq_street_diff"
    else: ac = "a_diff"
    return nc, ac
