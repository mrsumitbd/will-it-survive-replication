"""paper_numbers.py: map every empirical number of the paper to the value reproduced by the package.

Usage:
  python src/analysis/paper_numbers.py                      compare results/ (written by reproduce.sh) with the paper
  python src/analysis/paper_numbers.py --results expected_results
                                                            compare the shipped outputs behind the paper
  python src/analysis/paper_numbers.py --results <dir> --out <file.md>

The paper's values are listed below, section by section, as they are printed in the paper. For each value the script
reads the reproduced value from the results directory, rounds it to the precision printed in the paper and reports
  match       the reproduced value rounds to the paper's value (or a stated condition holds, e.g. p < 0.001)
  mismatch    it does not (the reason is given where it is known)
  missing     the output that holds the value is not in the results directory
  partial     the value agrees, but the step cannot be reproduced independently from the shipped data (reason given)
  external    the number comes from an outside source (AIDev release, literature, a design choice); not counted
  pending     the number cannot be computed yet (Cohen's kappa: the second rater's labels are pending); not counted
Writes <results>/PAPER_NUMBERS.md (or --out) and prints the summary.
"""
import os, re, sys, json, math, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
ap.add_argument('--out', default=None)
args = ap.parse_args()
BASE = os.path.abspath(args.results)
OUT = args.out or os.path.join(BASE, 'PAPER_NUMBERS.md')
MODE = open(os.path.join(BASE, 'RUN_MODE.txt')).read().strip() if os.path.exists(os.path.join(BASE, 'RUN_MODE.txt')) else (
    'expected (shipped outputs of the paper run)' if os.path.basename(BASE) == 'expected_results' else 'unknown')

_cache = {}


def J(rel):
    if rel not in _cache:
        p = os.path.join(BASE, rel)
        _cache[rel] = json.load(open(p)) if os.path.exists(p) else None
    if _cache[rel] is None:
        raise FileNotFoundError(rel)
    return _cache[rel]


def CSV(rel):
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(rel)
    return pd.read_csv(p)


def TXT(rel):
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(rel)
    return open(p).read()


# ------------------------------------------------------------------------------------------------ accessors
def rq1(g='line'): return J(f'rq1/rq1_{g}.json')
def A1(v, g='line'): return rq1(g)[f'A1_{v}']
def H(h): return [h['hr'], h['lo'], h['hi']]
def W(v, k, g='line'): return H(A1(v, g)['within'][k])
def P(x): return 100 * x
def grp(v, g, k, w='mod_prw'): return P(A1(v, g)['groups'][k][w])
def A12(g, k): return rq1(g)['A12'][k]
def A5(a): return rq1('line')['A5'][a]
def misc(): return J('rq1/a_misc.json')
def a4(): return J('rq1/a4_ph_line.json')
def a7(): return J('rq1/a7_sub.json')
def a9(): return J('rq1/a9.json')['A9']
def c1(): return J('rq1/c1_calendar_line.json')
def rq2(): return J('rq2/rq2_fb.json')
def cs(): return J('cohort/cohort_stats.json')
def fun(i): return J('cohort/funnel.json')['rows'][i]
def aud(): return J('audit/audit_summary.json')
def A8(k): return misc()['A8'][k]
def r3(sub, f): return f'rq3/{sub}/results/{f}'
def lr(sub): return J(r3(sub, 'd1_lr_size_history.json'))
def tour_a(sub): return CSV(r3(sub, 'rq31_bow_tournament_summary_combined_count.csv')).set_index('model')
def tour_b(sub):
    t = CSV(r3(sub, 'rq32_process_tournament_summary.csv'))
    return t.pivot_table(index='model', columns='metric', values='mean')
def sk(sub, f):
    s = CSV(r3(sub, f)); return set(s[s['rank'] == 1].model)
def loc(sub='rq3a_agent'): return J(r3(sub, 'rq3a_line_localization_summary.json'))
def ex(sub): return J(r3(sub, f"{sub[:4]}_explain_summary.json"))


def strata_both():
    b = c1()['b_repo_x_month']
    if 'strata_with_agent_and_human' in b:
        return b['strata_with_agent_and_human']
    return J('rq1/c1_calendar_line_strata.json')['strata_with_agent_and_human']


def full_fix_log(sub):
    t = TXT('rq3/prep_log_full_fix.txt')
    m = re.search(rf'{sub}: (\d+) rows, dropped for missing PR features (\d+) \(([\d.]+)\)', t)
    return [int(m.group(2)), int(m.group(1)), 100 * float(m.group(3))]


def rq3b_bins(sub='rq3b_agent'):
    rep = ex(sub)['oos_report']; n = sum(rep[str(k)]['support'] for k in range(4))
    return [100 * rep[str(k)]['support'] / n for k in range(4)]


def random_macro_f1(sub='rq3b_agent'):
    """expected Macro F1 of a uniform random guess over the four bins, given the bin shares"""
    s = [x / 100 for x in rq3b_bins(sub)]
    return sum(2 * p * 0.25 / (p + 0.25) for p in s) / 4


def cochran(N=2092, z=1.959964, e=0.05, p=0.5):
    n0 = z * z * p * (1 - p) / (e * e); return n0 / (1 + (n0 - 1) / N)


# ------------------------------------------------------------------------------------------------ entries
E = []
SEC = ['']


def sec(name): SEC[0] = name


def n(where, what, paper, get, note=''):
    """numeric entry: every number in `paper` is compared with the matching value returned by `get`"""
    E.append(dict(sec=SEC[0], where=where, what=what, paper=paper, get=get, kind='num', note=note))


def cond(where, what, paper, get, note=''):
    """condition entry: `get` returns (bool, shown value)"""
    E.append(dict(sec=SEC[0], where=where, what=what, paper=paper, get=get, kind='cond', note=note))


def ext(where, what, paper, source, get=None):
    E.append(dict(sec=SEC[0], where=where, what=what, paper=paper, get=get, kind='external', note=source))


def pend(where, what, paper, note):
    E.append(dict(sec=SEC[0], where=where, what=what, paper=paper, get=None, kind='pending', note=note))


def partial(where, what, paper, get, note):
    E.append(dict(sec=SEC[0], where=where, what=what, paper=paper, get=get, kind='partial', note=note))


def approx(lo, hi):
    """for 'about x' statements: shown value(s) and whether all lie in [lo, hi]"""
    def f(vals):
        vals = vals if isinstance(vals, (list, tuple)) else [vals]
        return all(lo <= v <= hi for v in vals), ', '.join(f'{v:.4g}' for v in vals)
    return f


def hrfmt(h): return f'{h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}]'


def sig(h): return h['p'] < 0.05


# ---------------------------------------------------------------- Abstract
sec('Abstract')
n('Abstract', 'lines, PRs, repositories of the primary design', '648,959 lines, 2,794 PRs, 215 repositories',
  lambda: [A1('primary')['n'], A1('primary')['prs'], A1('primary')['repos']])
n('Abstract', 'agent HR, primary, with 95% CI', 'HR = 0.83 [0.75, 0.93]', lambda: W('primary', 'agent'))
n('Abstract', 'supervised-assistant HR', 'HR = 0.76', lambda: W('primary', 'supervised')[:1])
n('Abstract', 'Devin HR', 'HR = 0.95', lambda: W('primary', 'devin')[:1])
cond('Abstract', 'after mid-May 2025, supervised and Devin code modified later', 'both upper CI < 1',
     lambda: (c1()['a_all_after_cut']['within']['supervised']['hi'] < 1 and c1()['a_all_after_cut']['within']['devin']['hi'] < 1,
              f"sup {hrfmt(H(c1()['a_all_after_cut']['within']['supervised']))}, Devin {hrfmt(H(c1()['a_all_after_cut']['within']['devin']))}"))
cond('Abstract', 'without PR weighting the difference is not significant', 'agent p >= 0.05 (unweighted)',
     lambda: (A1('nv_unw')['within']['agent']['p'] >= 0.05, f"p = {A1('nv_unw')['within']['agent']['p']:.3f}"))
n('Abstract', "Cramer's V (RQ2)", "V = 0.080", lambda: [rq2()['primary_word']['cramers_v']])
n('Abstract', 'AUC-ROC size/history baseline vs. token model (agent files)', 'AUC-ROC 0.78 vs. 0.70',
  lambda: [lr('rq3a_agent')['auc_roc_mean'], tour_a('rq3a_agent').loc['RandomForest', 'auc_roc_mean']],
  'token model = random forest, the best model by AUC-PR')
n('Abstract', 'Macro F1 of RQ3b vs. random guess', 'Macro F1 of 0.31 against 0.25',
  lambda: [tour_b('rq3b_agent').loc['RandomForest', 'macro_f1'], random_macro_f1()],
  'first-rank models: XGBoost 0.316, CatBoost 0.313, random forest 0.312; random guess = expected Macro F1 of a uniform guess over the four bins')

# ---------------------------------------------------------------- Introduction
sec('1 Introduction')
ext('Sec. 1', 'agent PRs in AIDev', '932,791 PRs', 'AIDev release (Li et al.)', lambda: [J_aidev()['agent_prs']])
ext('Sec. 1', 'Stack Overflow survey 2024', '76% of respondents', 'literature')
ext('Sec. 1', 'GitClear changed lines', 'about 153 million changed lines from 2020 to 2023; less than two weeks', 'literature')
n('Sec. 1', 'mined cohort', '2,802 merged PRs, 1,470 agent-attributed and 1,332 human, in 215 repositories',
  lambda: [cs()['mined_cohort']['agent'] + cs()['mined_cohort']['human'], cs()['mined_cohort']['agent'], cs()['mined_cohort']['human'], cs()['mined_cohort']['repos']])
n('Sec. 1 (RQ1)', 'supervised-assistant HR', 'HR = 0.76 [0.65, 0.87]', lambda: W('primary', 'supervised'))
n('Sec. 1 (RQ1)', 'Devin HR', 'HR = 0.95', lambda: W('primary', 'devin')[:1])
n('Sec. 1 (RQ2)', "Cramer's V", "V = 0.080", lambda: [rq2()['primary_word']['cramers_v']])
n('Sec. 1 (RQ2)', 'corrective share agent vs. human', '35.6% vs. 32.5%',
  lambda: [rq2()['primary_word']['pct']['Corrective']['Agent'], rq2()['primary_word']['pct']['Corrective']['Human']])
n('Sec. 1 (RQ3)', 'token model AUC-ROC', 'AUC-ROC 0.70', lambda: [tour_a('rq3a_agent').loc['RandomForest', 'auc_roc_mean']])
n('Sec. 1 (RQ3)', 'baseline AUC-ROC', 'AUC-ROC 0.78', lambda: [lr('rq3a_agent')['auc_roc_mean']])
n('Sec. 1 (RQ3)', 'Macro F1 vs. chance', 'Macro F1 0.31 against 0.25',
  lambda: [tour_b('rq3b_agent').loc['RandomForest', 'macro_f1'], random_macro_f1()])


def J_aidev():
    return json.load(open(os.path.join(ROOT, 'data', 'aidev', 'aidev_summary.json')))


# ---------------------------------------------------------------- Methodology
sec('2.1 Dataset')
ext('Sec. 2.1.1', 'AIDev agent PRs per agent', '932,791: 814,522 Codex, 50,447 Copilot, 32,941 Cursor, 29,744 Devin, 5,137 Claude Code',
    'AIDev release; checked against data/aidev/aidev_summary.json',
    lambda: [J_aidev()['agent_prs']] + [J_aidev()['agent_prs_by_agent'][a] for a in ['OpenAI_Codex', 'Copilot', 'Cursor', 'Devin', 'Claude_Code']])
ext('Sec. 2.1.1', 'AIDev repositories', '116,211 repositories', 'AIDev release (distinct repo_id); data/aidev/aidev_summary.json',
    lambda: [J_aidev()['agent_repos_by_repo_id']])
ext('Sec. 2.1.1', 'AIDev collection end', 'up to August 1, 2025', 'AIDev release')
ext('Sec. 2.1.1', 'AIDev human PRs (created between January and June 2025)', '6,618 human PRs in 818 repositories',
    'AIDev release; data/aidev/aidev_summary.json', lambda: [J_aidev()['human_prs'], J_aidev()['human_repos_by_slug']])
sec('Table 1 (funnel)')
ext('Table 1', 'AIDev, all repositories', '116,211 | 932,791 | 6,618', 'AIDev release; data/aidev/aidev_summary.json',
    lambda: [fun(0)['repos'], fun(0)['agent_prs'], fun(0)['human_prs']])
for i, (lab, pv) in enumerate([('(1) repositories with agent and human PRs', '816 | 9,797 | 6,598'), ('(2) software license', '666 | 7,625 | 5,054'),
                               ('(3) not archived, release or tag, >=2 contributors', '566 | 6,891 | 4,559'),
                               ('(4) bottom-quartile removal per language', '266 | 4,649 | 3,196')], start=1):
    n('Table 1', lab, pv, (lambda i=i: [fun(i)['repos'], fun(i)['agent_prs'], fun(i)['human_prs']]))
partial('Table 1', '(5) code ratio within the 97% interval', '256 | 4,563 | 3,132', lambda: [fun(5)['repos'], fun(5)['agent_prs'], fun(5)['human_prs']],
        'the per-repository code ratios of the final run are not available, so step (5) is reproduced from the cohort list '
        '(246 repositories with merged PRs + 10 Q1 repositories without any merged PR); the 10 removed repositories are inferred')
n('Table 1', 'repositories with at least one merged PR', '246 | 2,302 | 2,386', lambda: [cs()['merged_in_256'][k] for k in ['repos', 'agent', 'human']])
n('Table 1', 'repository history available', '244 | 2,266 | 2,351', lambda: [cs()['history_available'][k] for k in ['repos', 'agent', 'human']])
n('Table 1', 'not listed in both AIDev tables', '244 | 2,261 | 2,346', lambda: [cs()['not_in_both_tables'][k] for k in ['repos', 'agent', 'human']])
n('Table 1', 'merge commit located', '241 | 2,092 | 2,156', lambda: [cs()['merge_located'][k] for k in ['repos', 'agent', 'human']])
n('Table 1', 'adds at least one tracked source line (mined cohort)', '215 | 1,470 | 1,332', lambda: [cs()['mined_cohort'][k] for k in ['repos', 'agent', 'human']])
ext('Table 1 note', 'human PRs in repositories without agent PRs', '20 in two (2) repositories', 'AIDev release; data/aidev/aidev_summary.json',
    lambda: [J_aidev()['human_prs_in_repos_without_agent_prs'], J_aidev()['human_repos_without_agent_prs']])
sec('2.1.3 Final cohort')
n('Sec. 2.1.3', 'filtered repositories, with a merged PR', '256 filtered repositories, 246 have at least one merged PR',
  lambda: [fun(5)['repos'], cs()['merged_in_256']['repos']])
cond('Sec. 2.1.3', 'two repositories cannot be mined', 'dust-tt/dust and elizaos/eliza',
     lambda: (cs()['history_unavailable_repos'] == ['dust-tt/dust', 'elizaos/eliza'], ', '.join(cs()['history_unavailable_repos'])))
n('Sec. 2.1.3', 'PRs listed as both agent and human PRs', 'five (5) PRs', lambda: [cs()['ambiguous_prs']])
n('Sec. 2.1.3', 'merge commit not located', '359 PRs', lambda: [cs()['merge_not_found']])
n('Sec. 2.1.3', 'PRs that change no tracked source file', '1,446 PRs', lambda: [cs()['no_tracked_source_line']])
n('Sec. 2.1.3', 'mined cohort', '2,802 merged PRs in 215 repositories',
  lambda: [cs()['mined_cohort']['agent'] + cs()['mined_cohort']['human'], cs()['mined_cohort']['repos']])
n('Sec. 2.1.3', 'agent PRs by agent', '1,470 (492 Copilot, 491 Codex, 412 Devin, 44 Cursor, 31 Claude Code) and 1,332 human',
  lambda: [cs()['mined_cohort']['agent']] + [cs()['mined_cohort_by_agent'][a] for a in ['Copilot', 'OpenAI_Codex', 'Devin', 'Cursor', 'Claude_Code']] + [cs()['mined_cohort']['human']])
n('Sec. 2.1.3', 'PRs with only vendored or generated lines; primary design PRs', 'Eight (8) PRs; 2,794 PRs',
  lambda: [cs()['vendored_only_prs'], cs()['primary_prs']['agent'] + cs()['primary_prs']['human']])
sec('2.1.5 Attribution validation')
n('Sec. 2.1.5', 'merged agent PRs with a located merge commit', '2,092', lambda: [cs()['merge_located']['agent']])
n('Sec. 2.1.5', 'audited PRs (data/audit/audit_rater1.csv)', '325 PRs', lambda: [aud()['all']['n']])
cond('Sec. 2.1.5', 'sample size for a proportion, 95% confidence, +-5% margin, finite-population correction (N = 2,092)', '325',
     lambda: (math.ceil(cochran()) == 325, f'{cochran():.2f}, rounded up {math.ceil(cochran())}'))
n('Sec. 2.1.5', 'PRs in the second-rater sheet', '100 of the 325 PRs (31%)',
  lambda: [J('audit/kappa.json')['second_rater_rows'], aud()['all']['n'], 100 * J('audit/kappa.json')['second_rater_rows'] / aud()['all']['n']])
ext('Sec. 2.1.5', 'typical share for agreement; PRs needed', '10 to 25%; 62 PRs; kappa = 0.8 vs. 0.5', 'literature')
n('Sec. 2.1.5', "Cohen's kappa between the two raters", "kappa = 0.85", lambda: [J('audit/kappa.json')['cohen_kappa']])
n('Sec. 2.1.5', 'agent label', '218 (67.1% [61.8, 72.0])', lambda: [aud()['all']['agent'][0]] + [100 * x for x in aud()['all']['agent'][1:]])
n('Sec. 2.1.5', 'mixed, human, unclear labels', '71 (21.8%), 20 (6.2%), 16 (4.9%)',
  lambda: [aud()['all']['mixed'][0], 100 * aud()['all']['mixed'][1], aud()['all']['human'][0], 100 * aud()['all']['human'][1], aud()['all']['unclear'][0], 100 * aud()['all']['unclear'][1]])
n('Sec. 2.1.5', 'agent made substantive changes (agent or mixed)', '88.9%', lambda: [100 * aud()['all']['agent_or_mixed'][1]])
cond('Sec. 2.1.5', 'none of the 20 human PRs comes from Devin', 'Devin human labels = 0',
     lambda: (aud()['human_labelled_by_agent'].get('Devin', 0) == 0, str(aud()['human_labelled_by_agent'])))
n('Sec. 2.1.5', 'audited PRs in the mined cohort', '221 of the 325', lambda: [A8('in_mined'), A8('audited')])
n('Sec. 2.1.5', 'agent made substantive changes, mined cohort', '87.3% [82.3, 91.1]', lambda: [100 * x for x in A8('mined|all')['agent_or_mixed'][1:]])
n('Sec. 2.1.5', 'Devin', '98.4% of the 64 Devin PRs', lambda: [100 * A8('mined|Devin (autonomous)')['agent_or_mixed'][1], A8('mined|Devin (autonomous)')['n']])
n('Sec. 2.1.5', 'supervised assistants', '82.8% [76.1, 87.9] of the 157', lambda: [100 * x for x in A8('mined|Supervised assistants')['agent_or_mixed'][1:]] + [A8('mined|Supervised assistants')['n']])
n('Sec. 2.1.5', 'labeled human, mined cohort', '16 PRs (7.2% [4.5, 11.4])', lambda: [A8('mined|all')['human'][0]] + [100 * x for x in A8('mined|all')['human'][1:]])

sec('2.2 Measuring survival')
ext('Sec. 2.2.1', 'end of the observation window', 'December 31, 2025', 'design choice')
cond('Sec. 2.2.1', 'every unit is observed for at least five months', 'minimum follow-up >= 150 days',
     lambda: (cs()['min_followup_days'] >= 150, f"last merge {cs()['last_merge'][:10]}, minimum follow-up {cs()['min_followup_days']:.1f} days"))
ext('Sec. 2.2.3', 'substantial modification thresholds', 'theta in {25%, 50%, 75%, 100%}', 'design choice')
cond('Sec. 2.2.5', 'largest human PR', 'adds more than 34,000 lines outside vendored and generated paths',
     lambda: (cs()['largest_pr_lines_non_vendored']['Human'] > 34000, f"{cs()['largest_pr_lines_non_vendored']['Human']:,} lines outside vendored paths"))
n('Sec. 2.2.5', 'illustrative example (not data)', '10,000 of the 10,990 human lines (91%); one of the 100 PRs (1%)',
  lambda: [10000, 10990, 100 * 10000 / 10990, 100, 1.0], 'arithmetic of a hypothetical example')
sec('Table 2 (units and events)')
n('Table 2', 'units', '648,959 | 60,095 | 18,120', lambda: [A1('primary', g)['n'] for g in ['line', 'hunk', 'file']])
for lab, k in [('modified, human', 'Human'), ('modified, supervised assistants', 'Supervised assistants'), ('modified, Devin', 'Devin (autonomous)')]:
    pv = {'Human': '34.5% (30.1%) | 44.8% (39.3%) | 88.2% (85.2%)', 'Supervised assistants': '24.1% (23.3%) | 34.5% (30.9%) | 81.3% (77.1%)',
          'Devin (autonomous)': '38.0% (23.6%) | 52.5% (51.1%) | 91.5% (85.6%)'}[k]
    n('Table 2', lab, pv, (lambda k=k: sum([[P(A12(g, k)['mod_prw']), P(A12(g, k)['mod_unw'])] for g in ['line', 'hunk', 'file']], [])))
n('Table 2', 'median days to modification (modified units), PR-weighted (unweighted)', '40.1 (39.0) | 35.9 (40.1) | 10.0 (11.6)',
  lambda: sum([[A12(g, 'All')['median_days_to_mod_prw'], A12(g, 'All')['median_days_to_mod']] for g in ['line', 'hunk', 'file']], []))
n('Table 2', 'median follow-up of censored units (days)', '218.3 | 236.5 | 216.5', lambda: [A12(g, 'All')['median_followup_censored'] for g in ['line', 'hunk', 'file']])
n('Table 2 note', 'PRs and lines of the primary design', '2,794 PRs; 648,959 lines', lambda: [A1('primary')['prs'], A1('primary')['n']])
n('Table 2 note', "Devin's largest PR", 'adds 41.6% of its lines', lambda: [100 * A5('Devin')['largest_pr_share']])

# ---------------------------------------------------------------- RQ1
sec('3 RQ1')
ext('Sec. 3.2', 'RMST horizons and bootstrap', 'tau = 90 and 150 days; 1,000 repository bootstrap replicates', 'design choice; replicates read from results/rq1/rq1_line.json',
    lambda: [90, 150, rq1()['A3']['B']])
ext('Sec. 3.2', 'sampled design', 'at most 20 source files per PR and 100 random lines per file', 'design choice')
n('Sec. 3.3', 'agent HR, primary', 'HR = 0.83 [0.75, 0.93], p = 0.001', lambda: W('primary', 'agent') + [A1('primary')['within']['agent']['p']])
n('Sec. 3.3', 'supervised-assistant HR', 'HR = 0.76 [0.65, 0.87]', lambda: W('primary', 'supervised'))
cond('Sec. 3.3', 'supervised-assistant p', 'p < 0.001', lambda: (A1('primary')['within']['supervised']['p'] < 0.001, f"{A1('primary')['within']['supervised']['p']:.2g}"))
n('Sec. 3.3', 'Devin HR', 'HR = 0.95 [0.82, 1.11], p = 0.53', lambda: W('primary', 'devin') + [A1('primary')['within']['devin']['p']])
n('Sec. 3.3', 'hazard of supervised-assistant lines lower by', '24% lower', lambda: [100 * (1 - A1('primary')['within']['supervised']['hr'])])
n('Sec. 3.3', 'modified by the end of the window, agent vs. human (PR-weighted)', '28.0% and 34.5%',
  lambda: [rq1()['A3']['mod_agent_pct']['est'], rq1()['A3']['mod_human_pct']['est']])
n('Sec. 3.3', 'difference in percentage points [bootstrap CI]', '-6.5 pp [-10.7, -1.7]',
  lambda: [rq1()['A3']['mod_diff_pp'][k] for k in ['est', 'lo', 'hi']])
sec('Table 3 (RQ1 designs)')
T3 = [('Line, primary', 'primary', 'line', 'mod_prw', '648,959 | 28.0 / 34.5 | 0.83 [0.75, 0.93] | 0.76 [0.65, 0.87] | 0.95 [0.82, 1.11]'),
      ('Line, with vendored paths', 'all_prw', 'line', 'mod_prw', '854,139 | 28.1 / 34.8 | 0.83 [0.75, 0.93] | 0.76 [0.66, 0.87] | 0.95 [0.82, 1.10]'),
      ('Line, unweighted', 'nv_unw', 'line', 'mod_unw', '648,959 | 23.4 / 30.1 | 0.91 [0.69, 1.20] | 0.80 [0.60, 1.07] | 1.06 [0.65, 1.71]'),
      ('Line, unweighted, with vendored', 'all_unw', 'line', 'mod_unw', '854,139 | 23.3 / 50.3 | 0.92 [0.68, 1.25] | 0.86 [0.58, 1.29] | 1.00 [0.68, 1.48]'),
      ('Line, sampled', 'sampled_nv_prw', 'line', 'mod_prw', '277,283 | 28.1 / 34.7 | 0.83 [0.75, 0.93] | 0.76 [0.66, 0.88] | 0.95 [0.81, 1.10]'),
      ('Hunk, primary', 'primary', 'hunk', 'mod_prw', '60,095 | 39.5 / 44.8 | 0.93 [0.85, 1.02] | 0.82 [0.72, 0.92] | 1.11 [0.98, 1.26]'),
      ('File, primary', 'primary', 'file', 'mod_prw', '18,120 | 84.2 / 88.2 | 0.97 [0.90, 1.04] | 0.95 [0.86, 1.04] | 1.01 [0.90, 1.13]')]
for lab, v, g, w, pv in T3:
    n('Table 3', lab, pv, (lambda v=v, g=g, w=w: [A1(v, g)['n'], grp(v, g, 'Agent (all)', w), grp(v, g, 'Human', w)] + W(v, 'agent', g) + W(v, 'supervised', g) + W(v, 'devin', g)))
BOLD = {('Line, primary', 'agent'), ('Line, primary', 'supervised'), ('Line, with vendored paths', 'agent'), ('Line, with vendored paths', 'supervised'),
        ('Line, sampled', 'agent'), ('Line, sampled', 'supervised'), ('Hunk, primary', 'supervised')}


def bold_check():
    got = {(lab, k) for lab, v, g, w, _ in T3 for k in ['agent', 'supervised', 'devin'] if sig(A1(v, g)['within'][k])}
    return got == BOLD, 'same cells' if got == BOLD else f'p < 0.05 in {sorted(got ^ BOLD)} differs'


cond('Table 3', 'bold cells', 'bold exactly where p < 0.05', bold_check)
sec('Figure 1 (KM curves)')
cond('Figure 1', 'figure file', 'results/figures/rq1_km_line_primary.pdf',
     lambda: (os.path.exists(os.path.join(BASE, 'figures', 'rq1_km_line_primary.pdf')), 'present' if os.path.exists(os.path.join(BASE, 'figures', 'rq1_km_line_primary.pdf')) else 'absent'))
n('Figure 1 caption', 'no supervised-assistant line at risk after', 'day 281', lambda: [misc()['A13']['info']['Supervised assistants']['last_day_at_risk']])
n('Figure 1 caption', 'largest steps (supervised assistants, Devin)', 'day 223 and day 313',
  lambda: [misc()['A13']['info']['Supervised assistants']['largest_step_day'], misc()['A13']['info']['Devin (autonomous)']['largest_step_day']])
cond('Figure 1 caption', 'each step comes from a single PR', 'share of the step from its top PR = 1',
     lambda: (all(abs(misc()['A13']['info'][g]['largest_step_top_pr_share'] - 1) < 1e-9 for g in ['Supervised assistants', 'Devin (autonomous)']),
              ', '.join(f"{misc()['A13']['info'][g]['largest_step_top_pr_share']:.3f}" for g in ['Supervised assistants', 'Devin (autonomous)'])))
sec('3.3 Weighting, vendored code and pooling')
n('Sec. 3.3', 'share of lines in vendored paths: human, supervised, Devin', '34.1%, 3.8%, 0.3%',
  lambda: [P(misc()['A11'][g]['share_vendored_all_lines']) for g in ['Human', 'Supervised assistants', 'Devin (autonomous)']])
cond('Sec. 3.3', 'without weighting, no difference is significant', 'all unweighted line HRs p >= 0.05',
     lambda: (all(not sig(A1(v)['within'][k]) for v in ['nv_unw', 'all_unw'] for k in ['agent', 'supervised', 'devin']),
              ', '.join(f"{A1(v)['within'][k]['p']:.2f}" for v in ['nv_unw', 'all_unw'] for k in ['agent', 'supervised', 'devin'])))
n('Sec. 3.3', 'pooled Cox, primary design', 'HR = 0.89 [0.80, 1.00]', lambda: H(rq1()['A2_primary']))
n('Sec. 3.3', 'pooled Cox, unweighted lines with vendored paths', '0.76 [0.52, 1.10]', lambda: H(rq1()['A2_all_unw']))
sec('3.3 Proportional hazards and RMST')
cond('Sec. 3.3', 'Schoenfeld test (unweighted model)', 'p < 0.001', lambda: (a4()['agent_unweighted']['is_agent']['p'] < 0.001, f"{a4()['agent_unweighted']['is_agent']['p']:.2g}"),
     'the weighted test is not usable (fractional weights are treated as frequencies); the unweighted model test is reported')
n('Sec. 3.3', 'agent HR in days 0-14, 14-90, >90', '1.01 [0.83, 1.23], 0.85 [0.72, 1.01], 0.65 [0.53, 0.80]',
  lambda: sum([H(a4()['time_split'][f'agent_{p}']['is_agent']) for p in ['0_14', '14_90', '90_9999']], []))
n('Sec. 3.3', 'supervised HR in days 0-14, 14-90, >90', '1.10 [0.84, 1.44], 0.79 [0.64, 0.99], 0.47 [0.36, 0.61]',
  lambda: sum([H(a4()['time_split'][f'tier_{p}']['supervised']) for p in ['0_14', '14_90', '90_9999']], []))
cond('Sec. 3.3', 'Devin does not differ in any period', 'Devin p >= 0.05 in every period',
     lambda: (all(not sig(a4()['time_split'][f'tier_{p}']['devin']) for p in ['0_14', '14_90', '90_9999']),
              ', '.join(f"{a4()['time_split'][f'tier_{p}']['devin']['p']:.2f}" for p in ['0_14', '14_90', '90_9999'])))
n('Sec. 3.3', 'RMST difference at 90 days', '+0.8 days [-2.3, 3.8]', lambda: [rq1()['A3']['rmst90_diff'][k] for k in ['est', 'lo', 'hi']])
n('Sec. 3.3', 'RMST difference at 150 days', '+2.2 days [-3.4, 7.9]', lambda: [rq1()['A3']['rmst150_diff'][k] for k in ['est', 'lo', 'hi']])
n('Sec. 3.3', 'modified within 14 days (PR-weighted): supervised, Devin, human', '8.3%, 12.0%, 9.6%',
  lambda: [P(rq1()['A6'][g]['prw']) for g in ['Supervised assistants', 'Devin (autonomous)', 'Human']])
sec('3.3 Calendar time')
cond('Sec. 3.3', 'median merge dates', 'June 18, 2025 (supervised) vs. April 9 (human and Devin)',
     lambda: ([misc()['A11'][g]['merged_median'] for g in ['Supervised assistants', 'Human', 'Devin (autonomous)']] == ['2025-06-18', '2025-04-09', '2025-04-09'],
              ', '.join(misc()['A11'][g]['merged_median'] for g in ['Supervised assistants', 'Human', 'Devin (autonomous)'])))
ext('Sec. 3.3', 'calendar cut', 'May 16, 2025 (release of Codex)', 'design choice (product documentation)')
n('Sec. 3.3', 'PRs merged on or after May 16: agent, supervised, Devin HR', '0.70 [0.57, 0.84], 0.73 [0.58, 0.92], 0.61 [0.44, 0.84]',
  lambda: sum([H(c1()['a_all_after_cut']['within'][k]) for k in ['agent', 'supervised', 'devin']], []))
n('Sec. 3.3', 'Devin PRs after the cut', '133 Devin PRs', lambda: [c1()['a_all_after_cut']['groups']['Devin (autonomous)']['prs']])
n('Sec. 3.3', 'strata repository x merge month: supervised HR', '0.96 [0.74, 1.24]', lambda: H(c1()['b_repo_x_month']['supervised']))
n('Sec. 3.3', 'strata with agent and human PRs', '128 of the 657 strata', lambda: [strata_both(), c1()['b_repo_x_month']['strata']])
sec('3.3 Hunks, files and substantial modification')
n('Sec. 3.3', 'supervised-assistant hunks', 'HR = 0.82', lambda: W('primary', 'supervised', 'hunk')[:1])
cond('Sec. 3.3', 'agent group and Devin hunks not significant', 'p >= 0.05',
     lambda: (not sig(A1('primary', 'hunk')['within']['agent']) and not sig(A1('primary', 'hunk')['within']['devin']),
              f"{A1('primary', 'hunk')['within']['agent']['p']:.3f}, {A1('primary', 'hunk')['within']['devin']['p']:.3f}"))
n('Sec. 3.3', 'Devin hunks, unweighted', 'HR = 1.24 [1.00, 1.53], p = 0.049', lambda: W('nv_unw', 'devin', 'hunk') + [A1('nv_unw', 'hunk')['within']['devin']['p']])
cond('Sec. 3.3', 'at the file level, no group differs', 'all file HRs p >= 0.05',
     lambda: (all(not sig(A1('primary', 'file')['within'][k]) for k in ['agent', 'supervised', 'devin']), ', '.join(f"{A1('primary', 'file')['within'][k]['p']:.2f}" for k in ['agent', 'supervised', 'devin'])))
SUB = [f'{u}|{t}' for u in ['hunk', 'file'] for t in [25, 50, 75, 100]]
n('Sec. 3.3', 'substantial modification: agent HR range', 'HR 0.80 to 0.86', lambda: [min(a7()[s]['within']['agent']['hr'] for s in SUB), max(a7()[s]['within']['agent']['hr'] for s in SUB)])
cond('Sec. 3.3', 'substantial modification: agent p', 'all p < 0.02', lambda: (all(a7()[s]['within']['agent']['p'] < 0.02 for s in SUB), f"max p {max(a7()[s]['within']['agent']['p'] for s in SUB):.4f}"))
n('Sec. 3.3', 'substantial modification: supervised HR range', '0.72 to 0.77', lambda: [min(a7()[s]['within']['supervised']['hr'] for s in SUB), max(a7()[s]['within']['supervised']['hr'] for s in SUB)])
n('Sec. 3.3', 'substantial modification: Devin HR range', '0.93 to 1.01', lambda: [min(a7()[s]['within']['devin']['hr'] for s in SUB), max(a7()[s]['within']['devin']['hr'] for s in SUB)])
cond('Sec. 3.3', 'Devin does not differ at any theta', 'Devin p >= 0.05', lambda: (all(not sig(a7()[s]['within']['devin']) for s in SUB), f"min p {min(a7()[s]['within']['devin']['p'] for s in SUB):.3f}"))
sec('3.3 Attribution, file history and repository activity')
n('Sec. 3.3', 'cohort PRs labeled human; human or unclear', '16; 28', lambda: [A8('mined|all')['human'][0], A8('mined|all')['human'][0] + A8('mined|all')['unclear'][0]])
n('Sec. 3.3', 'agent HR without them', '0.84 and 0.83', lambda: [A8('rq1_excl_human')['within']['agent']['hr'], A8('rq1_excl_human_unclear')['within']['agent']['hr']])
n('Sec. 3.3', 'PRs labeled agent; HR', '136 audited PRs; 0.89 [0.64, 1.23]',
  lambda: [A8('rq1_agent_label_only_vs_all_human')['groups']['Agent (all)']['prs']] + H(A8('rq1_agent_label_only_vs_all_human')['within']['agent']))
n('Sec. 3.3', 'with file-history covariates: agent, supervised HR', '0.88 [0.79, 0.98]; 0.81 [0.70, 0.93]',
  lambda: H(a9()['line']['with_file_covariates']['agent']) + H(a9()['line']['with_file_covariates']['supervised']))


def quarter():
    b, c = a9()['line']['base_same_rows']['agent']['hr'], a9()['line']['with_file_covariates']['agent']['hr']
    return [(c - b) / (1 - b), (math.log(c) - math.log(b)) / -math.log(b)]


cond('Sec. 3.3 and 5.1', 'share of the difference explained by the files', 'about a quarter',
     lambda: (all(0.2 <= x <= 0.33 for x in quarter()), ', '.join(f'{100 * x:.0f}%' for x in quarter()) + ' (1 - HR scale, log scale)'))
n('Sec. 3.3', 'covariate HRs: recent changes, test file', 'HR = 1.26 per log unit; HR = 0.73',
  lambda: [a9()['line']['covariate_hrs']['prior_mod_3m']['hr'], a9()['line']['covariate_hrs']['test']['hr']])
n('Sec. 3.3', 'activity terciles (AIDev PR count): agent HR', '0.83 [0.44, 1.57], 0.63 [0.43, 0.94], 0.86 [0.77, 0.97]',
  lambda: sum([H(misc()['A10'][f'total_pr_count|{t}']['within']['agent']) for t in ['low', 'mid', 'high']], []))
cond('Sec. 3.3', 'same direction in every tercile', 'all HR < 1', lambda: (all(misc()['A10'][f'total_pr_count|{t}']['within']['agent']['hr'] < 1 for t in ['low', 'mid', 'high']), 'yes'))
sec('3.3 Individual agents')
n('Sec. 3.3', 'Copilot and Codex HR', 'HR = 0.75; HR = 0.69', lambda: [A5('Copilot')['hr']['hr'], A5('OpenAI_Codex')['hr']['hr']])
n('Sec. 3.3', 'PRs of Copilot and Codex', '980 of their 1,055 PRs', lambda: [A5('Copilot')['prs'] + A5('OpenAI_Codex')['prs'], A1('primary')['groups']['Supervised assistants']['prs']])
n('Sec. 3.3', 'PRs of Cursor and Claude Code', '44 and 31 PRs', lambda: [A5('Cursor')['prs'], A5('Claude_Code')['prs']])
n('Sec. 3.3', 'share of lines added by the largest PR, Cursor and Claude Code', '26% and 33%',
  lambda: [100 * A5(a)['largest_pr_share'] for a in ['Cursor', 'Claude_Code']])
sec('Table 4 (per agent)')
T4 = [('GitHub Copilot', 'Copilot', '489 | 0.75 [0.59, 0.94] | 22.5 [19.6, 25.3] | 9.7%'), ('OpenAI Codex', 'OpenAI_Codex', '491 | 0.69 [0.57, 0.84] | 24.4 [21.4, 27.8] | 5.7%'),
      ('Cursor', 'Cursor', '44 | 1.81 [0.91, 3.60] | 35.0 [24.6, 46.7] | 26.4%'), ('Claude Code', 'Claude_Code', '31 | 1.66 [0.81, 3.41] | 30.5 [18.0, 43.5] | 32.9%'),
      ('Devin', 'Devin', '412 | 0.96 [0.82, 1.11] | 38.0 [34.4, 41.5] | 41.6%')]
for lab, a, pv in T4:
    n('Table 4', lab, pv, (lambda a=a: [A5(a)['prs']] + H(A5(a)['hr']) + [P(A5(a)['mod_prw']), P(A5(a)['mod_prw_lo']), P(A5(a)['mod_prw_hi']), P(A5(a)['largest_pr_share'])]))
n('Table 4', 'Human', '1,327 | 34.5 [32.6, 36.6] | 9.0%', lambda: [A5('Human')['prs'], P(A5('Human')['mod_prw']), P(A5('Human')['mod_prw_lo']), P(A5('Human')['mod_prw_hi']), P(A5('Human')['largest_pr_share'])])
cond('Table 4', 'bold cells', 'bold exactly where p < 0.05 (Copilot, Codex)',
     lambda: ({a for _, a, _ in T4 if sig(A5(a)['hr'])} == {'Copilot', 'OpenAI_Codex'}, ', '.join(f"{a} p={A5(a)['hr']['p']:.3g}" for _, a, _ in T4)))

# ---------------------------------------------------------------- RQ2
sec('4 RQ2')
n('Sec. 4.2', 'merge commits among the modifying commits; their observations in the adaptive class', '416 merge commits; 280 of their 456 observations',
  lambda: [rq2()['merge_commits']['distinct_commits'], rq2()['merge_commits']['merge_intent']['Adaptive'], rq2()['merge_commits']['units']])
n('Sec. 4.3', 'observations', '6,112 (2,347 agent, 3,765 human)', lambda: [rq2()['primary_word'][k] for k in ['n', 'n_agent', 'n_human']])
n('Sec. 4.3', 'chi-square test', 'chi2 = 39.1, df = 4', lambda: [rq2()['primary_word']['chi2'], rq2()['primary_word']['dof']])
cond('Sec. 4.3', 'p-value', 'p < 0.001', lambda: (rq2()['primary_word']['p'] < 0.001, f"{rq2()['primary_word']['p']:.2g}"))
n('Sec. 4.3', "Cramer's V", 'V = 0.080', lambda: [rq2()['primary_word']['cramers_v']])
cond('Sec. 4.3', 'only the Other class deviates beyond |z| > 2', 'Other only',
     lambda: ({k for k, v in rq2()['primary_word']['std_resid']['Agent'].items() if abs(v) > 2} == {'Other'}, str({k: round(v, 2) for k, v in rq2()['primary_word']['std_resid']['Agent'].items()})))
n('Sec. 4.3', 'Other share; z', '10.6% vs. 15.9%, z = -4.28',
  lambda: [rq2()['primary_word']['pct']['Other']['Agent'], rq2()['primary_word']['pct']['Other']['Human'], rq2()['primary_word']['std_resid']['Agent']['Other']])
n('Sec. 4.3', 'Corrective share; z', '35.6% vs. 32.5%; z = +1.59',
  lambda: [rq2()['primary_word']['pct']['Corrective']['Agent'], rq2()['primary_word']['pct']['Corrective']['Human'], rq2()['primary_word']['std_resid']['Agent']['Corrective']])
n('Sec. 4.3', 'Perfective share', '38.3% vs. 35.4%', lambda: [rq2()['primary_word']['pct']['Perfective'][a] for a in ['Agent', 'Human']])
sec('Table 5 (RQ2 intent)')
for cl, pv in [('Corrective', '35.6 | 32.5 | +3.1 | +1.59'), ('Perfective', '38.3 | 35.4 | +2.9 | +1.46'), ('Adaptive', '11.7 | 12.7 | -1.0 | -0.89'),
               ('Preventive', '3.9 | 3.5 | +0.4 | +0.54'), ('Other', '10.6 | 15.9 | -5.3 | -4.28')]:
    n('Table 5', cl, pv, (lambda cl=cl: [rq2()['primary_word']['pct'][cl]['Agent'], rq2()['primary_word']['pct'][cl]['Human'],
                                        half_up(rq2()['primary_word']['pct'][cl]['Agent'], 1) - half_up(rq2()['primary_word']['pct'][cl]['Human'], 1), rq2()['primary_word']['std_resid']['Agent'][cl]]),
      'Delta = difference of the rounded shares, as in the paper')
n('Table 5 note', 'test statistics', 'chi2 = 39.1, df = 4, V = 0.080', lambda: [rq2()['primary_word']['chi2'], rq2()['primary_word']['dof'], rq2()['primary_word']['cramers_v']])
sec('4.3 Sensitivity analyses')
n('Sec. 4.3', 'without merge commits', 'V = 0.081', lambda: [rq2()['primary_word_no_merge']['cramers_v']])
n('Sec. 4.3', 'without merge commits: adaptive share (agent, human)', '8.3% (agent) and 8.4% (human)',
  lambda: [rq2()['primary_word_no_merge']['pct']['Adaptive']['Agent'], rq2()['primary_word_no_merge']['pct']['Adaptive']['Human']])
n('Sec. 4.3', 'single-group commits', 'V = 0.084, Corrective 36.2% vs. 32.4%',
  lambda: [rq2()['primary_word_single_group_commits']['cramers_v'], rq2()['primary_word_single_group_commits']['pct']['Corrective']['Agent'], rq2()['primary_word_single_group_commits']['pct']['Corrective']['Human']])
n('Sec. 4.3', 'file-commit unit', 'chi2 = 7.0, p = 0.13, V = 0.035', lambda: [rq2()['file_commit_word'][k] for k in ['chi2', 'p', 'cramers_v']])
n('Sec. 4.3', 'substring matching: effect size', 'V = 0.155', lambda: [rq2()['primary_substring']['cramers_v']])
n('Sec. 4.3', 'substring matching: preventive share agent, human', '29.7% and 18.8%', lambda: [rq2()['primary_substring']['pct']['Preventive'][a] for a in ['Agent', 'Human']])
n('Sec. 4.3', 'whole-word matching: preventive share agent, human', '3.9% and 3.5%', lambda: [rq2()['primary_word']['pct']['Preventive'][a] for a in ['Agent', 'Human']])

# ---------------------------------------------------------------- RQ3
sec('5 RQ3 approach')
cond('Sec. 5.2', '30 repetitions of 10-fold cross-validation', '300 scores per model',
     lambda: ((tour_a('rq3a_agent').n_scores == 300).all() and (CSV(r3('rq3b_agent', 'rq32_process_tournament_summary.csv')).n_scores == 300).all(),
              f"RQ3a {sorted(set(tour_a('rq3a_agent').n_scores))}, RQ3b {sorted(set(CSV(r3('rq3b_agent', 'rq32_process_tournament_summary.csv')).n_scores))}"))
ext('Sec. 5.2', 'explanation folds; vocabulary', 'five repository folds; at most 1,000 features; unigrams to trigrams', 'design choice')
n('Sec. 5.2 (RQ3b)', 'smallest bin', 'the smallest bin holds 15% of the files', lambda: [min(rq3b_bins())])
n('Sec. 5.2 (RQ3b)', 'files dropped for missing author history', '22 of the 6,300 modified agent files, 0.3%', lambda: full_fix_log('rq3b_agent'))
cond('Sec. 5.2 (RQ3b)', 'chance level of Macro F1 with four bins', '0.25', lambda: (round(random_macro_f1(), 2) == 0.25, f'{random_macro_f1():.4f}'))
sec('5.3 RQ3a results')
n('Sec. 5.3', 'agent dataset', '6,661 files from 155 repositories, 79.7% modified',
  lambda: [lr('rq3a_agent')['n'], ex('rq3a_agent')['repos'], 100 * lr('rq3a_agent')['pos_rate']])
cond('Sec. 5.3', 'first Scott-Knott rank (AUC-PR)', 'random forest, CatBoost and XGBoost',
     lambda: (sk('rq3a_agent', 'rq3a_sk_auc_pr.csv') == {'RandomForest', 'CatBoost', 'XGBoost'}, ', '.join(sorted(sk('rq3a_agent', 'rq3a_sk_auc_pr.csv')))))
cond('Sec. 5.3', 'first-rank AUC-ROC', 'about 0.70', lambda: approx(0.685, 0.715)([tour_a('rq3a_agent').loc[m, 'auc_roc_mean'] for m in ['RandomForest', 'CatBoost', 'XGBoost']]))
cond('Sec. 5.3', 'first-rank AUC-PR', 'about 0.89', lambda: approx(0.875, 0.905)([tour_a('rq3a_agent').loc[m, 'auc_pr_mean'] for m in ['RandomForest', 'CatBoost', 'XGBoost']]))
n('Sec. 5.3', 'AUC-PR of a random classifier', 'about 80% of the files are modified', lambda: [100 * lr('rq3a_agent')['pos_rate']])
n('Sec. 5.3', 'logistic regression baseline', 'AUC-ROC 0.78, AUC-PR 0.93', lambda: [lr('rq3a_agent')['auc_roc_mean'], lr('rq3a_agent')['auc_pr_mean']])
n('Sec. 5.3', 'all files', '15,946 files (83.8% modified)', lambda: [lr('rq3a_all')['n'], 100 * lr('rq3a_all')['pos_rate']])
cond('Sec. 5.3', 'all files: random forest, XGBoost, CatBoost AUC-ROC', 'about 0.69', lambda: approx(0.68, 0.70)([tour_a('rq3a_all').loc[m, 'auc_roc_mean'] for m in ['RandomForest', 'XGBoost', 'CatBoost']]))
n('Sec. 5.3', 'all files: baseline AUC-ROC', '0.75', lambda: [lr('rq3a_all')['auc_roc_mean']])
cond('Sec. 5.3', 'tokens with the largest SHAP values', 'key, model, config, error, state',
     lambda: ([t['token'] for t in ex('rq3a_agent')['shap_top_tokens'][:5]] == ['key', 'model', 'config', 'error', 'state'], ', '.join(t['token'] for t in ex('rq3a_agent')['shap_top_tokens'][:5])))
n('Sec. 5.3', 'LIME top tokens among the 50 highest SHAP', '46.9%', lambda: [100 * ex('rq3a_agent')['lime_tokens_in_shap_top50']])
n('Sec. 5.3', 'LIME runtime per file', 'about 0.2 seconds', lambda: [J(r3('rq3a_agent', 'lime_runtime.json'))['seconds_per_explanation']], 'machine dependent')
sec('5.3 RQ3a line localization')
n('Sec. 5.3', 'localization data', '1,941 files from 109 repositories; 163,409 lines, 22.8% modified',
  lambda: [loc()['files'], loc()['repos'], loc()['lines'], 100 * loc()['line_mod_rate']])
n('Sec. 5.3', 'SHAP ranking AUC-ROC (pooled, per-file mean); line length', 'AUC-ROC 0.546, per-file mean 0.576; length 0.552',
  lambda: [loc()['pooled_auc_roc'], loc()['mean_file_auc_roc'], loc()['length_baseline_pooled_auc_roc']])
n('Sec. 5.3', 'precision in the top 10% of lines: SHAP, random, longest lines', '36.7%, 30.1%, 39.7%',
  lambda: [100 * loc()['precision_at_10pct'], 100 * loc()['base_rate_mean_per_file'], 100 * loc()['precision_at_10pct_length_baseline']])
sec('5.3 RQ3b results')
n('Sec. 5.3', 'agent dataset', '6,278 modified files from 160 repositories', lambda: [ex('rq3b_agent')['n'], ex('rq3b_agent')['repos']])
n('Sec. 5.3', 'bins', '15.1% within one day, 25.0% within one week, 26.8% within one month, 33.1% later', lambda: rq3b_bins())
n('Sec. 5.3', 'AutoSpearman keeps', 'all eight (8) features', lambda: [len(J(r3('rq3b_agent', 'rq32_process_selected_features.json'))['selected_features'])])
cond('Sec. 5.3', 'first Scott-Knott rank (Macro F1)', 'XGBoost, CatBoost and random forest',
     lambda: (sk('rq3b_agent', 'rq3b_sk_macro_f1.csv') == {'XGBoost', 'CatBoost', 'RandomForest'}, ', '.join(sorted(sk('rq3b_agent', 'rq3b_sk_macro_f1.csv')))))
cond('Sec. 5.3', 'first-rank Macro F1', 'about 0.31', lambda: approx(0.30, 0.32)([tour_b('rq3b_agent').loc[m, 'macro_f1'] for m in ['XGBoost', 'CatBoost', 'RandomForest']]))
cond('Sec. 5.3', 'first-rank macro AUC-ROC', 'about 0.60', lambda: approx(0.59, 0.61)([tour_b('rq3b_agent').loc[m, 'auc_roc'] for m in ['XGBoost', 'CatBoost', 'RandomForest']]))
n('Sec. 5.3', 'out of sample, XGBoost', '0.32 and 0.61', lambda: [ex('rq3b_agent')['oos']['macro_f1'], ex('rq3b_agent')['oos']['auc_roc']])
cond('Sec. 5.3', 'all files: first Scott-Knott rank (Macro F1)', 'CatBoost alone',
     lambda: (sk('rq3b_all', 'rq3b_sk_macro_f1.csv') == {'CatBoost'}, ', '.join(sorted(sk('rq3b_all', 'rq3b_sk_macro_f1.csv')))))
n('Sec. 5.3', 'all files: CatBoost', 'Macro F1 0.32, AUC-ROC 0.62', lambda: [tour_b('rq3b_all').loc['CatBoost', 'macro_f1'], tour_b('rq3b_all').loc['CatBoost', 'auc_roc']])


def shap_order():
    return list(ex('rq3b_agent')['shap_overall'])


cond('Sec. 5.3', 'SHAP order', 'file changes (3 months) first, open PRs second, file age third',
     lambda: (shap_order()[:3] == ['file_modification_frequency_3m', 'project_open_prs_at_birth', 'file_age_days'], ', '.join(shap_order()[:3])))
cond('Sec. 5.3', 'LIME agrees on the leading feature', 'file changes (3 months)',
     lambda: (list(ex('rq3b_agent')['lime_overall'])[0] == 'file_modification_frequency_3m', list(ex('rq3b_agent')['lime_overall'])[0]))
cond('Sec. 5.3', 'PR-author history ranks', 'fifth and sixth of the eight features',
     lambda: (sorted([shap_order().index('contributor_prior_prs'), shap_order().index('contributor_acceptance_rate')]) == [4, 5],
              f"{shap_order().index('contributor_prior_prs') + 1} and {shap_order().index('contributor_acceptance_rate') + 1}"))
n('RQ3 summary box', 'token model vs. baseline AUC-ROC; Macro F1 vs. chance', '0.70; 0.78; 0.31 against 0.25',
  lambda: [tour_a('rq3a_agent').loc['RandomForest', 'auc_roc_mean'], lr('rq3a_agent')['auc_roc_mean'], tour_b('rq3b_agent').loc['RandomForest', 'macro_f1'], random_macro_f1()])

# ---------------------------------------------------------------- Summary boxes, discussion, threats, conclusion
sec('Summary boxes, Discussion, Threats, Conclusion')
n('RQ1 summary box', 'supervised-assistant HR', 'HR = 0.76', lambda: W('primary', 'supervised')[:1])
n('RQ2 summary box', "Cramer's V; corrective shares", 'V = 0.080; 35.6% vs. 32.5%',
  lambda: [rq2()['primary_word']['cramers_v'], rq2()['primary_word']['pct']['Corrective']['Agent'], rq2()['primary_word']['pct']['Corrective']['Human']])
cond('Sec. 6.1', 'difference largest after 90 days', 'HR lowest in the >90-day period',
     lambda: (min(['0_14', '14_90', '90_9999'], key=lambda p: a4()['time_split'][f'agent_{p}']['is_agent']['hr']) == '90_9999', 'yes'))
n('Sec. 6.2', 'supervised-assistant vs. human lines in test files (PR-weighted)', '35.6% vs. 24.1%',
  lambda: [P(a9()['line']['desc']['Supervised assistants']['test']), P(a9()['line']['desc']['Human']['test'])])
n('Sec. 6.5', 'Devin vs. human lines modified within two weeks', '12.0% vs. 9.6%', lambda: [P(rq1()['A6'][g]['prw']) for g in ['Devin (autonomous)', 'Human']])
n('Sec. 7', 'agent made substantive changes in the mined cohort', '87.3%', lambda: [100 * A8('mined|all')['agent_or_mixed'][1]])
n('Sec. 7', "second rater, Cohen's kappa", '100 PRs, kappa = 0.85', lambda: [J('audit/kappa.json')['joined'], J('audit/kappa.json')['cohen_kappa']])
n('Sec. 7', 'merge commit not located', '359 merged PRs', lambda: [cs()['merge_not_found']])
cond('Sec. 7', 'Cursor and Claude Code contribute fewer than 50 PRs each', '< 50', lambda: (A5('Cursor')['prs'] < 50 and A5('Claude_Code')['prs'] < 50, f"{A5('Cursor')['prs']}, {A5('Claude_Code')['prs']}"))
n('Sec. 7', 'repositories', '215 repositories', lambda: [A1('primary')['repos']])
ext('Sec. 8', 'related work', 'Xia and Miller: 182 repositories, HR = 1.11; Takerngsaksiri et al.: 3.68% vs. 2.34% within 30 days; '
    'Popescu et al.: three days, one week, three weeks; Kraishan: 90 days', 'literature')
n('Sec. 9', 'lines, PRs, repositories', '648,959 lines, 2,794 PRs, 215 repositories', lambda: [A1('primary')['n'], A1('primary')['prs'], A1('primary')['repos']])

# ------------------------------------------------------------------------------------------------ evaluation
NUM = re.compile(r'(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?')


def paper_numbers(s):
    out = []
    for m in NUM.finditer(s):
        t = m.group().replace('−', '-').replace(',', '')
        dec = len(t.split('.')[1]) if '.' in t else 0
        out.append((float(t), dec))
    return out


def same(paper, dec, v):
    return abs(v - paper) <= 0.5 * 10 ** (-dec) + 1e-9


def half_up(v, dec):
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(v)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


def double_rounded(paper, dec, v):
    """the paper's value equals the value rounded first to one more decimal and then half-up to the printed precision
    (e.g. HR 0.8347 -> 0.835 -> 0.84)"""
    return abs(half_up(half_up(v, dec + 1), dec) - paper) < 1e-9


def fmt(v, dec):
    return f'{v:,.{max(dec + 2, 2)}f}' if dec else (f'{v:,.0f}' if abs(v - round(v)) < 1e-9 else f'{v:,.3f}')


def esc(x): return str(x).replace('|', '\\|')


rows = []
for e in E:
    r = dict(sec=e['sec'], where=e['where'], what=e['what'], paper=e['paper'], note=e['note'])
    try:
        if e['kind'] == 'num' or e['kind'] == 'partial' or (e['kind'] == 'external' and e['get']):
            vals = list(e['get']())
            pn = paper_numbers(e['paper'])
            if len(pn) != len(vals):
                raise ValueError(f'{len(pn)} paper numbers vs {len(vals)} values')
            ok = [same(p, d, float(v)) for (p, d), v in zip(pn, vals)]
            dflag = [not o and double_rounded(p, d, float(v)) for (p, d), v, o in zip(pn, vals, ok)]
            if any(dflag):
                r['double'] = ', '.join(f'{p:g} from {float(v):.4f}' for (p, d), v, x in zip(pn, vals, dflag) if x)
                ok = [o or x for o, x in zip(ok, dflag)]
            r['repro'] = '; '.join(fmt(float(v), d) for (p, d), v in zip(pn, vals))
            r['n_nums'] = len(pn); r['n_ok'] = sum(ok)
            if e['kind'] == 'num':
                r['status'] = 'match' if all(ok) else 'mismatch'
                if not all(ok):
                    r['bad'] = ', '.join(f'{p:g} vs {fmt(float(v), d)}' for (p, d), v, o in zip(pn, vals, ok) if not o)
            elif e['kind'] == 'partial':
                r['status'] = 'partial' if all(ok) else 'mismatch'
            else:
                r['status'] = 'external'; r['note'] += ' (consistent)' if all(ok) else ' (NOT consistent with the shipped AIDev summary)'
        elif e['kind'] == 'cond':
            ok, shown = e['get']()
            r['repro'] = shown; r['status'] = 'match' if ok else 'mismatch'; r['n_nums'] = 1; r['n_ok'] = int(bool(ok))
        elif e['kind'] == 'external':
            r['repro'] = ''; r['status'] = 'external'
        else:
            r['repro'] = ''; r['status'] = 'pending'
    except FileNotFoundError as x:
        r['repro'] = f'missing output: {x}'; r['status'] = 'missing'
    except (KeyError, IndexError, TypeError, ValueError) as x:
        r['repro'] = f'error: {type(x).__name__}: {x}'; r['status'] = 'missing'
    rows.append(r)

cnt = pd.Series([r['status'] for r in rows]).value_counts().to_dict()
bad = [r for r in rows if r['status'] in ('mismatch', 'missing')]
dbl = [r for r in rows if r.get('double') and r['status'] == 'match']
cr = [r for r in rows if r['status'] in ('match', 'mismatch', 'missing', 'partial')]
NN = sum(r.get('n_nums', 1) for r in cr); NOK = sum(r.get('n_ok', 0) for r in cr if r['status'] in ('match', 'mismatch'))
counted = sum(cnt.get(k, 0) for k in ['match', 'mismatch', 'missing', 'partial'])
L = ['# Paper numbers: reproduced vs. reported', '']
if 'quick' in MODE:
    L += ['> **QUICK MODE (smoke test).** These results come from `bash reproduce.sh --quick` (RQ1 on 25% of the PRs, 50 bootstrap',
          '> replicates, RQ3 with 1 x 10 cross-validation, line localization on at most 40 files per fold). Most values are',
          '> NOT expected to match the paper. Run `bash reproduce.sh` for the paper settings, or compare the shipped outputs with',
          '> `python src/analysis/paper_numbers.py --results expected_results`.', '']
L += [f'Results directory: `{os.path.relpath(BASE, ROOT)}/` (mode: {MODE}).', '',
      f'**{NOK} of {NN} reproducible numbers match**, grouped in {counted} statements: **{cnt.get("match", 0)} of {counted} statements match** (mismatch: {cnt.get("mismatch", 0)}, missing: {cnt.get("missing", 0)}, '
      f'partial: {cnt.get("partial", 0)}; {len(dbl)} of the matches need double rounding, see below). Not counted: {cnt.get("external", 0)} external, {cnt.get("pending", 0)} pending.', '',
      'An entry groups the numbers of one statement; it matches when every number matches after rounding to the precision printed '
      'in the paper. Status: match, mismatch, missing (output absent), partial (agrees, but not independently reproducible from the shipped data), '
      'external (AIDev release, literature or a design choice), pending (second rater).', '']
if bad:
    L += ['## Mismatches and missing values', '', '| Where | What | Paper | Reproduced | Note |', '|---|---|---|---|---|']
    L += [f"| {r['where']} | {esc(r['what'])} | {esc(r['paper'])} | {esc(r['repro'])}{(' (' + r['bad'] + ')') if r.get('bad') else ''} | {esc(r['note'])} |" for r in bad] + ['']
if dbl:
    L += ['## Matches after double rounding', '',
          'These values match only when the reproduced value is first rounded to one more decimal and then rounded half-up to the '
          'printed precision (the paper rounds from 3-decimal summaries). Strict rounding gives the next lower value.', '',
          '| Where | What | Paper | Values |', '|---|---|---|---|']
    L += [f"| {r['where']} | {esc(r['what'])} | {esc(r['paper'])} | {r['double']} |" for r in dbl] + ['']
cur = None
for r in rows:
    if r['sec'] != cur:
        cur = r['sec']; L += ['', f'## {cur}', '', '| Where | What | Paper | Reproduced | Status | Note |', '|---|---|---|---|---|---|']
    st = r['status'] + (' (double rounding)' if r.get('double') and r['status'] == 'match' else '')
    L.append(f"| {r['where']} | {esc(r['what'])} | {esc(r['paper'])} | {esc(r['repro'])} | {st} | {esc(r['note'])} |")
open(OUT, 'w').write('\n'.join(L) + '\n')
for r in bad:
    print(f"{r['status'].upper():8s} {r['where']}: {r['what']}: paper {r['paper']} | reproduced {r['repro']} {r.get('bad', '')}")
print(f'{OUT}')
print(f'mode: {MODE}')
print(f"{len(dbl)} matching entries need double rounding (see the report)")
print(f"{NOK} / {NN} reproducible numbers match")
print(f"{cnt.get('match', 0)} / {counted} statements match (mismatch {cnt.get('mismatch', 0)}, missing {cnt.get('missing', 0)}, partial {cnt.get('partial', 0)}); "
      f"{cnt.get('external', 0)} external and {cnt.get('pending', 0)} pending not counted")
