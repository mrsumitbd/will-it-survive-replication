"""kappa.py: inter-rater agreement of the attribution audit (Section 2.1.5).

Joins the first-rater labels (data/audit/audit_rater1.csv) with the second-rater sheet
(data/audit/second_rater/second_rater_annotation_sheet.xlsx, sheet "annotate", column `label`) on the PR
(slug and pr_number) and computes the percent agreement, Cohen's kappa and Gwet's AC1 over the four labels
agent, mixed, human and unclear. Prints "second-rater labels not available yet" while the label column is empty.
Writes results/audit/kappa.json.
Usage: python src/analysis/kappa.py [<second_rater.xlsx or .csv>]
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import AUDIT, out
from xlsx_read import read_sheet

LABELS = ['agent', 'mixed', 'human', 'unclear']
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AUDIT, 'second_rater', 'second_rater_annotation_sheet.xlsx')
r2 = pd.read_csv(src, dtype=str) if src.endswith('.csv') else read_sheet(src, 'annotate')
r1 = pd.read_csv(os.path.join(AUDIT, 'audit_rater1.csv'), dtype=str)


def norm(d):
    d = d.copy()
    d['slug'] = d.slug.str.strip().str.lower()
    d['pr_number'] = pd.to_numeric(d.pr_number).astype(int)
    d['label'] = d.label.fillna('').str.strip().str.lower()
    return d[['slug', 'pr_number', 'label']]


r1, r2 = norm(r1), norm(r2)
res = {'second_rater_rows': int(len(r2)), 'second_rater_labelled': int((r2.label != '').sum())}
if res['second_rater_labelled'] == 0:
    res['status'] = 'second-rater labels not available yet'
    print('second-rater labels not available yet')
else:
    bad = sorted(set(r2.label[r2.label != '']) - set(LABELS))
    if bad:
        sys.exit(f'unknown second-rater labels: {bad} (expected one of {LABELS})')
    m = r1.merge(r2[r2.label != ''], on=['slug', 'pr_number'], suffixes=('_1', '_2'), how='inner')
    res['joined'] = int(len(m)); res['unlabelled_rows'] = int((r2.label == '').sum())
    po = float((m.label_1 == m.label_2).mean())
    p1 = m.label_1.value_counts(normalize=True); p2 = m.label_2.value_counts(normalize=True)
    pe = sum(p1.get(c, 0) * p2.get(c, 0) for c in LABELS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float('nan')
    pi = {c: (p1.get(c, 0) + p2.get(c, 0)) / 2 for c in LABELS}
    pe_g = sum(pi[c] * (1 - pi[c]) for c in LABELS) / (len(LABELS) - 1)
    res.update(status='computed', percent_agreement=po, cohen_kappa=float(kappa), gwet_ac1=float((po - pe_g) / (1 - pe_g)),
               confusion={a: {b: int(((m.label_1 == a) & (m.label_2 == b)).sum()) for b in LABELS} for a in LABELS})
    print(f"n = {len(m)} (unlabelled rows: {res['unlabelled_rows']}): agreement {po:.3f}, Cohen's kappa {kappa:.3f}, Gwet's AC1 {res['gwet_ac1']:.3f}")
    print(pd.crosstab(m.label_1.rename('rater 1'), m.label_2.rename('rater 2')).to_string())
json.dump(res, open(out('audit', 'kappa.json'), 'w'), indent=1)
