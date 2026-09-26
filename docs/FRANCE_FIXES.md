# France (unseen test country) — label-free diagnosis and normalisation fixes (HQ, 26 Sep)

France = 259,452 test S1 (15%), 703K S2, 732K S3 rows. No labels exist, so everything below was measured on
the raw French test files and on two kinds of **pseudo-pairs** built from them:

- **same-address pairs**: S2 rows whose exact normalised address matches exactly ONE S1 (210K pairs) → shows the
  *name* noise. On US/India training data the same construction is a true match 98.6% of the time, so it is a
  fair proxy.
- **same-name pairs**: S2/S3 rows whose exact folded name matches exactly ONE S1 (294K pairs) → shows the
  *address* noise.

## What the French vendor generator does (vs what our normalisation assumed)

| noise | share of FR S2/S3 | before | after the fix |
|---|---|---|---|
| `N° 32` / `Nº 32` / `No 32` / `(32)` / `# 32` house-number labels | 17% | `N°` folded to `n` → ADDR_CANON → **"north 32"** (5% of FR addresses contained a bogus `north`) | label stripped → `32` |
| zero-padded house numbers `0332`, `00262`, `01` | 3.3% (also **3.2% of US** and 5% of India S2/S3!) | kept → number mismatch | `332`, `262`, `1` (a 5-digit ZIP with one leading zero is kept) |
| region vs département: S1 always `…, Nantes, Pays de la Loire`; S2/S3 have the region (32%), the département `Loire-Atlantique` (32%) or nothing (36%) | 68% differ from S1 | 4 different strings | region + département → one code (`pdl`, `hdf`, `naq`, …; all 13 regions / 96 départements, matched only as a whole comma component or as a multi-word phrase) — the US analogue of `Illinois`→`il` |
| street-type abbreviations `R`/`R.` (12.7%), `Av`/`Ave`, `All.`, `Bd.`, `Imp.`, `Pl`, `Rte.`, `Ch`, `Crs`, `Q.` | ~25% | `r/av/all/bd/imp/pl/rte/ch` already mapped; `crs`, `q`, `chem`, `boulevrd` missing | added |
| `St-Nazaire`, `St Herblain` (city) | 0.8% | `st` → **street** | `st`/`ste` at the start of a digit-free comma component → `saint`/`sainte` (also fixes US `St. Louis`, India `St. John Road`) |
| `5BIS`, `22 Ter` | 4% | `ter` → **terrace** | `5 bis`; `ter` after a number kept |
| legal form moved to the front: `SARL JEUNE PHARMACIE`, `S.A.S. Monet` | 5.7% | prefix stripped but legal form LOST (`legal=''`) | recorded (`legal='sarl'`); `sasu`, `snc`, `scp`, `selarl` added as prefixes |
| `EI` (entreprise individuelle) — 7th most common S1 legal form | 1.6% of S1 | not a legal form → stayed in the core name, dropped in S2 → mismatch | `ei` legal form (suffix only) |
| `&` ↔ `et` (`Gym & Fils` ↔ `GYM ET FILS`) | 1.1% | `&`→`and`, `et` kept | `et`→`and` in fold (US/India: `et` as a word is ~0) |
| `Cie` ↔ `Compagnie` | 0.2% | different tokens | `cie`→`compagnie` |
| web-domain names with the legal form glued on: `folieclubsas.com` ↔ `Folie Club SAS` | 1.2% (35% of FR domain names; US/India ≈ 0%) | `folieclubsas` ≠ nospace `folieclub` | for domain/handle-derived single-token names only: trailing `sarl/sas/sasu/eurl/sci/snc/ei/inc/llc/ltd/…` split off → core `folieclub`, legal `sas` |
| social handles `@asseclub`, `#priveamis` (also US/India) | 0.3% | `@`→`at` → core **"at asseclub"** | leading `@`/`#` stripped |
| appended filler / swapped last word: `X Y SARL` → `X SARL & Fils` / `& Associés` / `Groupe` / `Développement` / `Participations` / `Holding` / `Distribution` / `International` / `Services` / `France` | ~13% | `extra_*_content` counted these as CONTENT words (the US list has `group/holdings/services/partners/…` but not the French words) → French pairs looked like a different entity | French filler + legal forms + articles (`de du des la le les`) added to the generic set in `features.py` |
| `(France)` dropped / added, word order swaps, accents deleted inside domain names (`coleprimaireeglise.com` = école primaire église) | 5% / 0.5% | handled by fold (parentheses) / not handled | unchanged (accent deletion: too rare to matter) |

Not present in France: 5-digit postcodes (0.5%), cedex, arrondissements. 36% of FR S2/S3 addresses carry neither
region nor département while S1 always does (US S2/S3 lack the state only 4.5%, India 14%) — this residual
"missing state" shift remains; a possible follow-up is imputing the region from the city using S1 itself.

## Effect on the pseudo-pairs (French test data, no labels)

| proxy | before | after |
|---|---|---|
| same-name pairs: normalised address identical | 14.2% | **32.5%** |
| same-name pairs: address token sets identical | 20.4% | **46.6%** |
| same-name pairs: mean address token Jaccard | 0.655 | **0.818** |
| same-name pairs: first house number equal | 81.3% | 84.3% |
| same-name pairs: bogus `north` in the S2/S3 address | 5.04% | 0.02% |
| same-address pairs: no-space core name identical | 57.8% | 60.2% |
| same-address pairs: S2 legal form detected | 46.3% | 54.6% |

US/India side effects (train slice, 300K records): `n_core` changes on 0.3% of records (almost all `@handle`
names), `legal` on 0.01–0.03%, `n_addr` on 3.6% (US) / 7.1% (India) — of which 90% are zero-padded house
numbers now stripped, the rest `St.`→`saint` cities and single letters no longer glued across a comma
(`Block A, B Wing` → `block a b wing` instead of `block ab wing`). Regression check on the 8% slice: see
`runs/slice_frnorm/` (HQ sandbox run, old HQ pipeline; compare with `runs/slice_v3`: OOF 0.9768 / holdout 0.9826).

## Files changed

- `src/normalize.py`: LEGAL_CANON (+ei/eirl/selas/sarlu/scm/scea, −as), NAME_PREFIXES (+sasu/snc/scp/selarl, legal
  recorded), NAME_TOKEN_CANON (et→and, cie→compagnie), `_handle`/`_is_compact` + `split_legal(compact=True)`,
  FR_REGIONS → `_FR_CANON`, ADDR_CANON (+crs/q/chem/fg/rle/prom/esp/res/boulevrd), `norm_addr` rewritten to work
  per comma component (number labels, leading zeros, bis/ter, region canon, st→saint), `_normalize_frame` passes
  the compact flag. Output columns unchanged.
- `src/features.py`: generic-token set extended (French filler/legal/articles + dba/fka/aka/honorifics).

## How to evaluate on the box (no labels for France)

1. Normalisation changed ⇒ the per-country stores must be rebuilt (`cache_*/*.parquet` + `.done.json`), for the
   slices and for the full data. Candidate sets built with the old normalisation stay valid candidate sets, so the
   fast lane (`--reuse-candidates`) can rescore v2's `candidate_pairs.tsv` with recomputed features + a retrained
   model — the France gain shows up mostly in the features, not in blocking.
2. Q (slice_mix / xc_us / xc_in) must not drop; the leading-zero fix should if anything help US/India.
3. France, label-free: on `output_vN/`, compare with v2 the France mean max-probability per S1, the predicted-empty
   rate (v2: 5.2%) and the share of predicted pairs whose addresses now agree on the region code; the same-name /
   same-address pseudo-pair agreement above can be recomputed from the new store columns.
