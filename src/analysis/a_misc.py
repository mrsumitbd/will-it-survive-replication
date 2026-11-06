"""a_misc.py [A8 A10 A11 A13]: attribution audit vs. cohort and RQ1 without audited PRs (A8), repository-activity terciles (A10),
descriptives per group (A11), Figure 1 KM curves (A13). Writes results/rq1/a_misc.json and results/figures/rq1_km_line_primary.{pdf,png}."""
import sys, os, json, math, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
which = sys.argv[1:] or ['A11', 'A13', 'A8', 'A10']
T = fbc.Timer(); d0 = fbc.load('line'); dp = fbc.variant(d0, 'primary', 'line')
path = fbc.out('rq1', 'a_misc.json'); res = json.load(open(path)) if os.path.exists(path) else {}
def save(): json.dump(res, open(path, 'w'), indent=1, default=float)
Z = 1.959964
def wilson(k, n):
    if n == 0: return [None] * 3
    p = k / n; dd = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / dd; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / dd
    return [p, c - h, c + h]

if 'A11' in which:
    TEST = r'(?:^|/)(?:test|tests|spec|specs|__tests__)/|test'
    d = d0.copy()
    d['test'] = d.file.str.lower().str.contains(r'(?:^|/)(?:test|tests|spec|__tests__)/') | d.file.str.lower().str.rsplit('/', n=1).str[-1].str.contains('test')
    f = fbc.load('file')
    # new file = file that did not exist at the PR base. Not in the tables -> approximate: all lines of the file at merge were added by this PR is unknown;
    # use file_added_lines vs. lines: not available. We report what the tables support (see SUMMARY note).
    nv = d[~d.vendored]
    a11 = {}
    for g, x in list(d.groupby('group')) + [('Agent (all)', d[d.is_agent == 1])]:
        xn = x[~x.vendored]
        w = 1 / xn.groupby(['slug', 'pr_number']).event.transform('size')
        a11[g] = dict(lines_all=int(len(x)), share_vendored_all_lines=float(x.vendored.mean()),
                      lines_nv=int(len(xn)), share_test_nv=float(xn.test.mean()), share_test_nv_prw=float(np.average(xn.test, weights=w)))
    p = pd.read_parquet(os.path.join(fbc.T, 'prs.parquet')); p = p[p.status == 'ok']
    keys = dp[['slug', 'pr_number']].drop_duplicates()
    p = p.merge(keys, on=['slug', 'pr_number']); p['m'] = pd.to_datetime(p.merged_at, utc=True)
    for g, x in list(p.groupby('group')) + [('Agent (all)', p[p.is_agent == 1])]:
        q = x.m.quantile([.25, .5, .75])
        a11[g].update(prs=int(len(x)), merged_q1=str(q.iloc[0].date()), merged_median=str(q.iloc[1].date()), merged_q3=str(q.iloc[2].date()),
                      merged_min=str(x.m.min().date()), merged_max=str(x.m.max().date()),
                      by_month={str(k): int(v) for k, v in x.m.dt.strftime('%Y-%m').value_counts().sort_index().items()})
    res['A11'] = a11; save(); T('A11')

if 'A13' in which:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cols = [('Human', '#1f77b4'), ('Supervised assistants', '#d62728'), ('Devin (autonomous)', '#2ca02c')]
    ticks = [0, 60, 120, 180, 240, 300, 360]; rows = []; info = {}
    for g, c in cols:
        x = dp[dp.group == g]
        k = KaplanMeierFitter(label=g).fit(x.duration_days, x.event, weights=x.w)
        k.plot_survival_function(ax=ax, color=c, ci_show=False, linewidth=1.6)
        rows.append((g, [int((x.duration_days >= t).sum()) for t in ticks]))
        info[g] = dict(max_followup_days=float(x.duration_days.max()), last_day_at_risk=float(x.duration_days.max()))
        sf = k.survival_function_.iloc[:, 0]; drop = (sf.shift(1) - sf).fillna(0)
        t0 = float(drop.idxmax()); info[g]['largest_step_day'] = t0; info[g]['largest_step_size'] = float(drop.max())
        ev = x[(x.event == 1) & (x.duration_days.round(6) == round(t0, 6))]
        top = ev.groupby(['slug', 'pr_number']).w.sum().sort_values(ascending=False)
        info[g]['largest_step_top_pr'] = str(top.index[0]) if len(top) else None
        info[g]['largest_step_top_pr_share'] = float(top.iloc[0] / top.sum()) if len(top) else None
        # unweighted largest step as well
        ku = KaplanMeierFitter().fit(x.duration_days, x.event); su = ku.survival_function_.iloc[:, 0]; du = (su.shift(1) - su).fillna(0)
        tu = float(du.idxmax()); evu = x[(x.event == 1) & (x.duration_days.round(6) == round(tu, 6))]
        tpu = evu.groupby(['slug', 'pr_number']).size().sort_values(ascending=False)
        info[g].update(unw_largest_step_day=tu, unw_largest_step_size=float(du.max()), unw_step_top_pr=str(tpu.index[0]) if len(tpu) else None,
                       unw_step_top_pr_lines=int(tpu.iloc[0]) if len(tpu) else None)
    ax.set_xlabel('Days since merge'); ax.set_ylabel('Probability that a line is unmodified (PR-weighted)')
    ax.set_xlim(0, 370); ax.set_ylim(0, 1.02); ax.legend(loc='lower left', frameon=False)
    hdr = 'Lines at risk, day'; W = max([len(hdr)] + [len(g) for g, _ in rows]) + 2
    txt = f'{hdr:<{W}}' + ''.join(f'{t:>8d}' for t in ticks) + '\n' + '\n'.join(f'{g:<{W}}' + ''.join(f'{v:>8d}' for v in r) for g, r in rows)
    ax.text(0.0, -0.2, txt, transform=ax.transAxes, fontsize=6.5, family='monospace', va='top')
    plt.tight_layout(); fig.savefig(fbc.out('figures', 'rq1_km_line_primary.pdf'), bbox_inches='tight'); fig.savefig(fbc.out('figures', 'rq1_km_line_primary.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    res['A13'] = dict(at_risk={g: dict(zip(map(str, ticks), r)) for g, r in rows}, info=info); save(); T('A13')

def sub_within(d, label):
    d = fbc.prw(d); r = fbc.counts(d); r['within'] = fbc.within(d); T(f'{label} {fbc.fmt(r["within"]["agent"])}'); return r

if 'A8' in which:
    a = pd.read_csv(os.path.join(fbc.AUDIT, 'audit_rater1.csv'), dtype=str); a['label'] = a.label.str.strip().str.lower(); a['pr_number'] = a.pr_number.astype(float).astype(int)
    a['slug'] = a.slug.str.lower()
    mined = d0[['slug', 'pr_number']].drop_duplicates().assign(slug=lambda x: x.slug.str.lower())
    prim = dp[['slug', 'pr_number']].drop_duplicates().assign(slug=lambda x: x.slug.str.lower())
    a['in_mined'] = a.set_index(['slug', 'pr_number']).index.isin(mined.set_index(['slug', 'pr_number']).index)
    a['in_primary'] = a.set_index(['slug', 'pr_number']).index.isin(prim.set_index(['slug', 'pr_number']).index)
    a8 = {'audited': int(len(a)), 'in_mined': int(a.in_mined.sum()), 'in_primary': int(a.in_primary.sum())}
    for scope, s in [('mined', a[a.in_mined]), ('primary', a[a.in_primary])]:
        for grp, x in [('all', s)] + list(s.groupby('group')):
            n = len(x); out = {'n': n}
            for lab in ['agent', 'mixed', 'human', 'unclear']:
                out[lab] = [int((x.label == lab).sum())] + wilson(int((x.label == lab).sum()), n)
            out['agent_or_mixed'] = [int(x.label.isin(['agent', 'mixed']).sum())] + wilson(int(x.label.isin(['agent', 'mixed']).sum()), n)
            a8[f'{scope}|{grp}'] = out
    a8['by_agent_mined'] = {k: v.label.value_counts().to_dict() for k, v in a[a.in_mined].groupby('agent_name')}
    res['A8'] = a8; save(); T('A8 counts')
    dl = d0[~d0.vendored].copy(); dl['k'] = list(zip(dl.slug.str.lower(), dl.pr_number))
    lab = dict(zip(zip(a.slug, a.pr_number), a.label))
    dl['lab'] = dl.k.map(lab)
    for name, keep in [('excl_human', ~dl.lab.isin(['human'])), ('excl_human_unclear', ~dl.lab.isin(['human', 'unclear'])),
                       ('agent_label_only_vs_all_human', (dl.is_agent == 0) | (dl.lab == 'agent')),
                       ('agent_or_mixed_label_vs_all_human', (dl.is_agent == 0) | dl.lab.isin(['agent', 'mixed']))]:
        res['A8'][f'rq1_{name}'] = sub_within(dl[keep].drop(columns=['k', 'lab']), name); save()

if 'A10' in which:
    m = pd.read_csv(os.path.join(fbc.COHORT, 'repo_meta_v6.csv'))[['slug', 'total_pr_count', 'open_issues', 'size_kb']]
    a10 = {}
    for metric in ['total_pr_count', 'open_issues']:
        rr = m[m.slug.isin(dp.slug.unique())].copy()
        rr['t'] = pd.qcut(rr[metric].rank(method='first'), 3, labels=['low', 'mid', 'high'])
        cut = rr.groupby('t')[metric].agg(['min', 'max', 'size']).to_dict('index')
        for t in ['low', 'mid', 'high']:
            s = dp[dp.slug.isin(rr[rr.t == t].slug)].drop(columns='w')
            r = sub_within(s, f'{metric} {t}'); r['range'] = cut[t]; a10[f'{metric}|{t}'] = r; res['A10'] = a10; save()
T('done')
