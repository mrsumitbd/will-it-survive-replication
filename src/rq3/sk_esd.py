"""
sk_esd.py: Scott-Knott ESD test (parametric version, the ScottKnottESD R package default: sk_esd(x, version="p")).

Port of klainfo/ScottKnottESD R/partition.R::PartitionParametric:
  * treatments sorted by mean (descending)
  * recursively split the current range at the cut that maximises the between-group sum of squares
  * a range is NOT split when Cohen's d between its first and last treatment is negligible (|d| < 0.2,
    effsize::cohen.d with pooled SD), so ranks differ only by a non-negligible effect.
Usage: python sk_esd.py <wide.csv> <out.csv>   (wide.csv: one column per model, one row per CV iteration)
"""
import sys
import numpy as np, pandas as pd


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return 0.0 if sp == 0 else (a.mean() - b.mean()) / sp


def negligible(a, b):
    if np.array_equal(np.asarray(a), np.asarray(b)):
        return True
    return abs(cohen_d(a, b)) < 0.2


def sk_esd(wide: pd.DataFrame) -> pd.Series:
    order = wide.mean().sort_values(ascending=False).index.tolist()
    means = wide.mean()[order].values
    groups = {}
    counter = [0]

    def part(lo, hi):  # inclusive range [lo, hi] of positions in `order`
        if lo == hi or negligible(wide[order[lo]], wide[order[hi]]):
            counter[0] += 1
            for i in range(lo, hi + 1):
                groups[order[i]] = counter[0]
            return
        best, cut = -np.inf, lo
        for k1 in range(lo, hi):
            t1, t2 = means[lo:k1 + 1].sum(), means[k1 + 1:hi + 1].sum()
            n1, n2 = k1 - lo + 1, hi - k1
            ss = t1 ** 2 / n1 + t2 ** 2 / n2 - (t1 + t2) ** 2 / (n1 + n2)
            if ss > best:
                best, cut = ss, k1
        part(lo, cut)
        part(cut + 1, hi)

    part(0, len(order) - 1)
    return pd.Series(groups)[order]


if __name__ == '__main__':
    wide = pd.read_csv(sys.argv[1])
    wide = wide.drop(columns=[c for c in wide.columns if c.lower() in ('iteration', 'rep', 'fold', 'unnamed: 0')], errors='ignore')
    r = sk_esd(wide)
    out = pd.DataFrame({'model': r.index, 'rank': r.values, 'mean': wide.mean()[r.index].values,
                        'median': wide.median()[r.index].values})
    out.to_csv(sys.argv[2], index=False)
    print(out.to_string(index=False))
