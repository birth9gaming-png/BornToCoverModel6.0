"""Colley win/loss rankings and neutralized Massey point ratings."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def fit(samples, home_field):
    names = sorted({name for away, home, *_ in samples for name in (away, home)})
    index = {name: i for i, name in enumerate(names)}
    n = len(names); colley = 2 * np.eye(n); b = np.ones(n)
    design = np.zeros((len(samples), n)); margin = np.zeros(len(samples))
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for row, (away, home, a, h, neutral) in enumerate(samples):
        ai, hi = index[away], index[home]
        colley[ai, ai] += 1; colley[hi, hi] += 1
        colley[ai, hi] -= 1; colley[hi, ai] -= 1
        outcome = np.sign(a - h)
        b[ai] += outcome / 2; b[hi] -= outcome / 2
        design[row, ai] = 1; design[row, hi] = -1
        margin[row] = a - h + (0 if neutral else home_field)
        parent[root(ai)] = root(hi)
    colley_r = np.linalg.solve(colley, b)
    # Minimum-norm least squares sets the mean within each disconnected
    # component to zero. Cross-component Massey differences are withheld.
    massey = np.linalg.lstsq(design, margin, rcond=None)[0]
    return dict(colley=dict(zip(names, colley_r.tolist())),
                massey=dict(zip(names, massey.tolist())),
                components={name: root(i) for name, i in index.items()})
