"""rq2_fb.py [--from-clones <dir>]: RQ2 on the primary units (non-vendored lines), word-boundary classifier,
unit = modifying commit x author group, with the sensitivity analyses (no merge commits, single-group commits,
substring classifier, file x commit unit). Writes results/rq2/rq2_fb.json.
Parent counts of the modifying commits (merge-commit check) come from data/rq2/commit_parents.csv.
With --from-clones <dir> they are recomputed from local bare clones (<owner>__<name>.git), as in the paper's run."""
import sys, os, json, re, numpy as np, pandas as pd
from scipy.stats import chi2_contingency
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, 'ease_src'))
sys.path.insert(0, os.path.dirname(H)); sys.path.insert(0, os.path.join(os.path.dirname(H), 'mining'))
from paths import TABLES, RQ2_DATA, out
from mine_death_details import CLASSIFICATION_RULES, classify_message
from extract_features import REASON_MAPPING, REASON_LABELS
T = TABLES
CL = sys.argv[sys.argv.index('--from-clones') + 1] if '--from-clones' in sys.argv else None
PRIORITY = ['Security', 'Corrective (Bug Fix)', 'Performance', 'Adaptive (New Feature)', 'Perfective (Refactor)',
            'Preventive (Testing)', 'Documentation', 'Chore/Maintenance']
WORD_RE = {c: [re.compile(r'(?<![a-z0-9])' + re.escape(k) + r'(?![a-z0-9])') for k in ks] for c, ks in CLASSIFICATION_RULES.items()}
def classify_word(msg):
    m = str(msg).lower(); hits = {c for c, regs in WORD_RE.items() if any(r.search(m) for r in regs)}
    for c in PRIORITY:
        if c in hits: return c
    return 'Other'
sw = lambda c: REASON_LABELS[REASON_MAPPING.get(c, 4)]
def test(df, col):
    ct = pd.crosstab(df.author_type, df[col]); chi2, p, dof, exp = chi2_contingency(ct)
    n = ct.values.sum(); v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))); resid = (ct - exp) / np.sqrt(exp)
    pct = (ct.T / ct.sum(1)).T * 100
    return dict(n=int(n), n_agent=int(ct.loc['Agent'].sum()), n_human=int(ct.loc['Human'].sum()), chi2=float(chi2), dof=int(dof), p=float(p), cramers_v=float(v),
                pct={k: {a: float(x) for a, x in v_.items()} for k, v_ in pct.to_dict().items()},
                std_resid={a: {k: float(x) for k, x in resid.loc[a].items()} for a in resid.index})
c = pd.read_parquet(os.path.join(T, 'commits.parquet'))
c['intent_word'] = c.message.apply(classify_word).map(sw); c['intent_substring'] = c.message.apply(lambda m: classify_message(m)[0]).map(sw)
l = pd.read_parquet(os.path.join(T, 'lines.parquet')); l = l[l.event == 1]
f = pd.read_parquet(os.path.join(T, 'files.parquet')); f = f[f.event == 1]
# parent counts of the modifying commits: shipped cache, or the local clones (same logic, src/mining/commit_parents.py)
if CL:
    from commit_parents import parents_from_clones
    par = parents_from_clones(pd.concat([l[['slug', 'death_sha']], f[['slug', 'death_sha']]]).drop_duplicates(), CL)
else:
    cp = pd.read_csv(os.path.join(RQ2_DATA, 'commit_parents.csv'))
    par = dict(zip(cp.sha, cp.n_parents))
c['n_parents'] = c.sha.map(par)
res = {'parents_found': int(c.n_parents.notna().sum()), 'commits': int(len(c))}
def unitize(d):
    return d.merge(c[['sha', 'intent_word', 'intent_substring', 'n_parents']], left_on='death_sha', right_on='sha').drop_duplicates(['death_sha', 'author_type'])
# reproduction of the earlier run (all lines, word, commit x group): expect n=6182, V=0.083
res['repro_all_lines_word'] = test(unitize(l), 'intent_word')
lp = l[~l.vendored]; u = unitize(lp)
res['primary_word'] = test(u, 'intent_word')
res['primary_substring'] = test(u, 'intent_substring')
res['merge_commits'] = dict(units=int((u.n_parents > 1).sum()), distinct_commits=int(u[u.n_parents > 1].death_sha.nunique()),
                            units_total=int(len(u)), parents_missing=int(u.n_parents.isna().sum()),
                            merge_by_group=u[u.n_parents > 1].author_type.value_counts().to_dict(),
                            merge_intent=u[u.n_parents > 1].intent_word.value_counts().to_dict())
res['primary_word_no_merge'] = test(u[u.n_parents <= 1], 'intent_word')
ng = lp.groupby('death_sha').author_type.nunique(); single = set(ng[ng == 1].index)
res['primary_word_single_group_commits'] = test(u[u.death_sha.isin(single)], 'intent_word')
res['commits_touching_both_groups'] = int((ng > 1).sum())
fu = unitize(f[~f.vendored]); res['file_commit_word'] = test(fu, 'intent_word'); res['file_commit_substring'] = test(fu, 'intent_substring')
res['classifier_agreement_commits'] = float((c.intent_word == c.intent_substring).mean())
json.dump(res, open(out('rq2', 'rq2_fb.json'), 'w'), indent=1)
for k, v in res.items():
    if isinstance(v, dict) and 'cramers_v' in v:
        print(k, v['n'], round(v['chi2'], 2), '%.3g' % v['p'], round(v['cramers_v'], 4), {a: round(x, 2) for a, x in v['pct'].get('Preventive', {}).items()})
    else: print(k, v)
