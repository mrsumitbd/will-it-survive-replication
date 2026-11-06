"""audit_summary.py: summary of the first-rater attribution audit (Section 2.1.5).

Reads data/audit/audit_rater1.csv (the "audit" sheet of attribution_audit_sheet_rater1.xlsx as CSV) and reports the
label counts with 95% Wilson score intervals (Brown, Cai and DasGupta 2001), overall, per tier and per agent.
The sample is proportional by agent, so the overall shares are self-weighting. The shares restricted to the mined
cohort are in results/rq1/a_misc.json (A8). Agreement with the second rater: src/analysis/kappa.py.
Writes results/audit/audit_summary.json and prints a text version."""
import os, sys, math, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import AUDIT, out

Z = 1.959964
LABELS = ['agent', 'mixed', 'human', 'unclear']


def wilson(k, n):
    if n == 0:
        return [float('nan')] * 3
    p = k / n; d = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [p, c - h, c + h]


r1 = pd.read_csv(os.path.join(AUDIT, 'audit_rater1.csv'), dtype=str)
r1['label'] = r1.label.str.strip().str.lower()
print('rows', len(r1), '| missing labels', int(r1.label.isna().sum()))
print(pd.crosstab(r1.agent_name, r1.label, margins=True).to_string())


def shares(d):
    n = len(d); o = {'n': n}
    for lab in LABELS:
        k = int((d.label == lab).sum()); o[lab] = [k] + wilson(k, n)
    k = int(d.label.isin(['agent', 'mixed']).sum()); o['agent_or_mixed'] = [k] + wilson(k, n)
    return o


res = {'all': shares(r1)}
res.update({f'group|{g}': shares(d) for g, d in r1.groupby('group')})
res.update({f'agent|{g}': shares(d) for g, d in r1.groupby('agent_name')})
res['human_labelled_by_agent'] = {k: int(v) for k, v in r1[r1.label == 'human'].agent_name.value_counts().items()}
json.dump(res, open(out('audit', 'audit_summary.json'), 'w'), indent=1)
for k, v in res.items():
    if isinstance(v, dict) and 'n' in v:
        print(f"{k:32s} n={v['n']:4d} " + ' '.join(f"{lab}={v[lab][0]} ({100 * v[lab][1]:.1f}% [{100 * v[lab][2]:.1f}, {100 * v[lab][3]:.1f}])" for lab in LABELS + ['agent_or_mixed']))
print('human-labelled PRs by agent:', res['human_labelled_by_agent'])
