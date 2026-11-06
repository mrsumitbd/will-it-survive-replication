"""fbc.py: shared helpers of the RQ1 analyses (loading, primary design, PR weights, Cox models, bootstrap)."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import TABLES, DATA, COHORT, AUDIT, FILE_COV, RQ2_DATA, out, quick
from lifelines import CoxPHFitter
T = TABLES
GROUPS = ['Supervised assistants', 'Devin (autonomous)', 'Human']
COV = ['n_added_lines', 'n_source_files', 'stars', 'contributors']
AGENTS = ['Copilot', 'OpenAI_Codex', 'Cursor', 'Claude_Code', 'Devin']
FN = {'line': 'lines', 'hunk': 'hunks', 'file': 'files'}


QUICK_SHARE = 0.25  # reproduce.sh --quick: share of PRs kept (smoke test only; the paper uses all PRs)


def load(gran, tables=T):
    d = pd.read_parquet(os.path.join(tables, FN[gran] + '.parquet'))
    d = d[d.event.notna()].copy(); d['event'] = d.event.astype(int)
    if quick():  # smoke test: a fixed random quarter of the PRs, so that every model runs in minutes
        prs = d[['slug', 'pr_number']].drop_duplicates()
        keep = prs.sample(frac=QUICK_SHARE, random_state=0)
        d = d.merge(keep, on=['slug', 'pr_number'])
    return d.reset_index(drop=True)


def prw(d):
    d = d.copy(); d['w'] = 1.0 / d.groupby(['slug', 'pr_number']).duration_days.transform('size'); return d


def ease_sample(dd, gran):
    files = dd[['slug', 'pr_number', 'file']].drop_duplicates().groupby(['slug', 'pr_number']).head(20)
    dd = dd.merge(files, on=['slug', 'pr_number', 'file'])
    if gran == 'line':
        dd['_r'] = np.random.default_rng(42).random(len(dd))
        dd = dd[dd.groupby(['slug', 'pr_number', 'file'])._r.rank(method='first') <= 100].drop(columns='_r')
    return dd


def variant(d, v, gran):
    """primary = non-vendored, PR-weighted (weights after the filter)."""
    if gran == 'file' and 'file' not in d.columns:
        d = d.assign(file=d.unit_id)
    if v == 'primary':
        return prw(d[~d.vendored])
    if v == 'all_prw':
        return prw(d)
    if v == 'nv_unw':
        return d[~d.vendored].assign(w=1.0)
    if v == 'all_unw':
        return d.assign(w=1.0)
    if v == 'sampled_nv_prw':
        return prw(ease_sample(d[~d.vendored], gran))
    raise ValueError(v)


def wrate(d):
    return float(np.average(d.event, weights=d.w))


def rmst_np(t, e, w, tau):
    """Weighted Kaplan-Meier RMST up to tau (same estimator as lifelines KaplanMeierFitter(weights) + restricted_mean_survival_time)."""
    o = np.argsort(t, kind='mergesort'); t, e, w = t[o], e[o], w[o]
    ut, idx = np.unique(t, return_index=True)
    wsum = np.add.reduceat(w, idx); dsum = np.add.reduceat(w * e, idx)
    atrisk = w.sum() - np.concatenate([[0], np.cumsum(wsum)[:-1]])
    s = np.cumprod(1 - np.where(atrisk > 0, dsum / atrisk, 0))
    m = ut < tau
    xs = np.concatenate([[0], ut[m], [tau]]); ss = np.concatenate([[1], s[m]])
    return float(np.sum(np.diff(xs) * ss))


class Boot:
    """Repository cluster bootstrap with numpy index arrays."""
    def __init__(self, d):
        self.d = d.reset_index(drop=True)
        codes, self.repos = pd.factorize(self.d.slug)
        o = np.argsort(codes, kind='mergesort'); self.order = o
        self.starts = np.searchsorted(codes[o], np.arange(len(self.repos)))
        self.ends = np.append(self.starts[1:], len(o))
        self.t = self.d.duration_days.to_numpy(float); self.e = self.d.event.to_numpy(float)
        self.w = self.d.w.to_numpy(float); self.a = self.d.is_agent.to_numpy(int)

    def stats(self, ix):
        t, e, w, a = self.t[ix], self.e[ix], self.w[ix], self.a[ix]
        A, H = a == 1, a == 0
        ra = np.average(e[A], weights=w[A]); rh = np.average(e[H], weights=w[H])
        out = [100 * ra, 100 * rh, 100 * (ra - rh)]
        for tau in (90, 150):
            x, y = rmst_np(t[A], e[A], w[A], tau), rmst_np(t[H], e[H], w[H], tau)
            out += [x, y, x - y]
        return out

    names = ['mod_agent_pct', 'mod_human_pct', 'mod_diff_pp', 'rmst90_agent', 'rmst90_human', 'rmst90_diff',
             'rmst150_agent', 'rmst150_human', 'rmst150_diff']

    def run(self, B, seed=0):
        rng = np.random.default_rng(seed); n = len(self.repos); vals = []
        for _ in range(B):
            pick = rng.integers(0, n, n)
            ix = np.concatenate([self.order[self.starts[i]:self.ends[i]] for i in pick])
            try:
                vals.append(self.stats(ix))
            except ZeroDivisionError:
                pass
        vals = np.array(vals)
        point = self.stats(np.arange(len(self.t)))
        return {k: dict(est=point[i], lo=float(np.percentile(vals[:, i], 2.5)), hi=float(np.percentile(vals[:, i], 97.5)))
                for i, k in enumerate(self.names)} | {'B_ok': int(len(vals))}


def _cox(x, dummies, strata=True, extra=()):
    cols = ['duration_days', 'event', 'w', 'cluster'] + list(dummies) + list(extra) + (['slug'] if strata else [])
    c = CoxPHFitter().fit(x[cols], 'duration_days', 'event', weights_col='w', cluster_col='cluster', robust=True,
                          strata=['slug'] if strata else None)
    return c, {k: dict(hr=float(np.exp(c.summary.loc[k].coef)), lo=float(c.summary.loc[k]['exp(coef) lower 95%']),
                       hi=float(c.summary.loc[k]['exp(coef) upper 95%']), p=float(c.summary.loc[k].p)) for k in dummies}


def design(d, extra_log=()):
    x = d.copy()
    for c in ['n_added_lines', 'n_source_files'] + list(extra_log):
        x[c] = np.log1p(x[c].astype(float))
    x['cluster'] = x.slug + '#' + x.pr_number.astype(str)
    x['devin'] = (x.group == 'Devin (autonomous)').astype(int); x['supervised'] = (x.group == 'Supervised assistants').astype(int)
    for a in AGENTS:
        x['ag_' + a] = (x.agent_name == a).astype(int)
    return x


def within(d, extra=(), extra_log=()):
    x = design(d, extra_log)
    base = ['n_added_lines', 'n_source_files'] + list(extra)
    _, a = _cox(x, ['is_agent'], extra=base)
    _, t = _cox(x, ['supervised', 'devin'], extra=base)
    return dict(agent=a['is_agent'], **t)


def pooled(d):
    x = design(d)
    for c in ['stars', 'contributors']:
        x[c] = np.log1p(x[c].astype(float))
    c, r = _cox(x, ['is_agent'], strata=False, extra=COV)
    r['is_agent']['concordance'] = float(c.concordance_index_); return r['is_agent']


def counts(d):
    out = dict(n=int(len(d)), prs=int(d[['slug', 'pr_number']].drop_duplicates().shape[0]), repos=int(d.slug.nunique()))
    g = {}
    for k, x in list(d.groupby('group')) + [('Agent (all)', d[d.is_agent == 1])]:
        g[k] = dict(n=int(len(x)), prs=int(x[['slug', 'pr_number']].drop_duplicates().shape[0]), repos=int(x.slug.nunique()),
                    mod_unw=float(x.event.mean()), mod_prw=float(np.average(x.event, weights=1 / x.groupby(['slug', 'pr_number']).event.transform('size'))))
    out['groups'] = g; return out


def logrank(d):
    from lifelines.statistics import logrank_test
    a, h = d[d.is_agent == 1], d[d.is_agent == 0]
    r = logrank_test(a.duration_days, h.duration_days, a.event, h.event, weights_A=a.w, weights_B=h.w)
    return dict(stat=float(r.test_statistic), p=float(r.p_value))


def fmt(h):
    return f"{h['hr']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}] p={h['p']:.3g}"


class Timer:
    def __init__(self): self.t = time.time()
    def __call__(self, msg): print(f'[{time.time() - self.t:7.1f}s] {msg}', flush=True)
