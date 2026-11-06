"""make_summary.py [<results_dir>]: write <results_dir>/SUMMARY.md, a long-form summary of every analysis output
(default results/; missing parts are marked PENDING). Run on expected_results/ to see the outputs behind the paper."""
import os, sys, json, glob, datetime, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS
H = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
def J(p):
    p = p if os.path.isabs(p) else f'{H}/{p}'
    return json.load(open(p)) if os.path.exists(p) else None
def hr(h): return f"{h['hr']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}], p={h['p']:.3g}" if h else 'n/a'
def pc(x): return f'{100 * x:.2f}%'
G3 = ['Supervised assistants', 'Devin (autonomous)', 'Human', 'Agent (all)']
L = []; w = L.append
w(f'# Analysis summary\n\nGenerated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} by `src/analysis/make_summary.py` from `{os.path.basename(H)}/`.\n')
w('**Primary design**: line level, full cohort (215 repositories, 2,794 PRs), vendored/generated paths excluded, every unit weighted 1/(units in its PR) with weights computed after the vendored filter, Cox model stratified by repository with robust SEs clustered by PR, covariates log(PR added lines) and log(PR source files). HR < 1 means agent-attributed code is modified later than human code.\n')
VN = {'primary': 'Primary (no vendored, PR-weighted)', 'all_prw': '(a) all lines, PR-weighted (old primary)', 'nv_unw': '(b) no vendored, unweighted',
      'all_unw': '(c) all lines, unweighted', 'sampled_nv_prw': '(d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted'}
R = {g: J(f'rq1/rq1_{g}.json') for g in ['line', 'hunk', 'file']}
w('## A. RQ1\n\n### A1. Primary design and sensitivity variants\n')
for g, r in R.items():
    if not r: w(f'{g}: PENDING\n'); continue
    w(f'#### {g.title()} level\n')
    w('| Variant | Units | PRs | Repos | Agent HR (within-repo) | Supervised HR | Devin HR | Log-rank p |\n|---|---|---|---|---|---|---|---|')
    for v, name in VN.items():
        x = r.get(f'A1_{v}')
        if x: w(f"| {name} | {x['n']:,} | {x['prs']:,} | {x['repos']} | {hr(x['within']['agent'])} | {hr(x['within']['supervised'])} | {hr(x['within']['devin'])} | {x['logrank']['p']:.3g} |")
        else: w(f'| {name} | PENDING |||||||')
    w('\nModification rates (unweighted / PR-weighted), PRs per group:\n')
    w('| Variant | ' + ' | '.join(G3) + ' |\n|---|' + '---|' * 4)
    for v, name in VN.items():
        x = r.get(f'A1_{v}')
        if x: w(f'| {name} | ' + ' | '.join(f"{pc(x['groups'][k]['mod_unw'])} / {pc(x['groups'][k]['mod_prw'])} ({x['groups'][k]['prs']:,} PRs, {x['groups'][k]['n']:,} units)" for k in G3) + ' |')
    w('')
w('### A2. Pooled Cox, no strata, covariates log PR added lines, files, stars, contributors, robust SEs clustered by PR\n')
w('| Level | Primary | All units, unweighted |\n|---|---|---|')
for g, r in R.items():
    if r: w(f"| {g} | {hr(r.get('A2_primary'))} | {hr(r.get('A2_all_unw'))} |")
pass  # w('\nNote: `scripts/rq1.py` fitted the pooled model with `robust=weighted`, so the unweighted pooled model had model-based SEs and no clustering (hence the narrow [0.75, 0.76] in the draft). Here every pooled model uses `cluster_col` = PR.\n')
w('### A3. RMST and modification-rate differences (primary design, repository cluster bootstrap)\n')
w('| Level | B | Mod. rate agent | Mod. rate human | Diff (pp) [95% CI] | RMST90 agent / human (days) | RMST90 diff [CI] | RMST150 agent / human | RMST150 diff [CI] |\n|---|---|---|---|---|---|---|---|---|')
for g, r in R.items():
    a = r and r.get('A3')
    if a:
        f = lambda k, d=4: f"{a[k]['est']:.{d}f} [{a[k]['lo']:.{d}f}, {a[k]['hi']:.{d}f}]"
        w(f"| {g} | {a['B']} ({a['B_ok']} ok) | {a['mod_agent_pct']['est']:.4f}% | {a['mod_human_pct']['est']:.4f}% | {f('mod_diff_pp')} | {a['rmst90_agent']['est']:.3f} / {a['rmst90_human']['est']:.3f} | {f('rmst90_diff', 3)} | {a['rmst150_agent']['est']:.3f} / {a['rmst150_human']['est']:.3f} | {f('rmst150_diff', 3)} |")
w('\nModification rates in A3 are PR-weighted (each PR counts once) and unrounded.\n')
a4 = J('rq1/a4_ph_line.json')
w('### A4. Proportional hazards (primary line model)\n')
if a4:
    w('Schoenfeld-residual test (lifelines `proportional_hazard_test`, rank transform):\n')
    for k, v in a4.items():
        if k == 'time_split': continue
        w(f'- {k}: ' + ('; '.join(f"{c} stat={x['stat']:.2f} p={x['p']:.3g}" for c, x in v.items()) if isinstance(v, dict) else v))
    w('\nThe weighted test statistics are not usable: lifelines treats the fractional PR weights as frequency weights in the residual test, which inflates the statistics to 10^5-10^6. Use the unweighted model test (same stratified model without weights and clustering). With 650k lines, any departure from PH is "significant"; the time-split HRs below show its size and direction.\n')
    w('Time-split within-repo HRs (primary design; separate stratified Cox fits on the units at risk at the start of each period):\n')
    w('| Period (days) | Agent | Supervised | Devin |\n|---|---|---|---|')
    ts = a4.get('time_split', {})
    for lo, hi in [('0', '14'), ('14', '90'), ('90', '9999')]:
        A = ts.get(f'agent_{lo}_{hi}', {}).get('is_agent'); T = ts.get(f'tier_{lo}_{hi}', {})
        w(f"| {lo}-{hi if hi != '9999' else 'end'} | {hr(A)} | {hr(T.get('supervised'))} | {hr(T.get('devin'))} |")
    w('')
else: w('PENDING\n')
w('### A5. Per-agent results (primary design)\n')
for g, r in R.items():
    a5 = r and r.get('A5')
    if not a5: continue
    w(f'#### {g.title()} level\n\n| Agent | Units | PRs | Repos | Within-repo HR vs Human | PR-weighted mod. rate [PR bootstrap 95% CI] | Unweighted rate | Share of units from largest PR (PR) |\n|---|---|---|---|---|---|---|---|')
    for k, x in a5.items():
        w(f"| {k} | {x['units']:,} | {x['prs']} | {x['repos']} | {hr(x['hr']) if x['hr'] else 'reference'} | {pc(x['mod_prw'])} [{pc(x['mod_prw_lo'])}, {pc(x['mod_prw_hi'])}] | {pc(x['mod_unw'])} | {pc(x['largest_pr_share'])} {x['largest_pr']} |")
    w('')
w('### A6. 14-day modification rate (primary design: non-vendored, PR-weighted)\n\n| Level | ' + ' | '.join(G3) + ' |\n|---|' + '---|' * 4)
for g, r in R.items():
    a6 = r and r.get('A6')
    if a6: w(f'| {g} | ' + ' | '.join(f"{pc(a6[k]['prw'])} (unweighted {pc(a6[k]['unw'])})" for k in G3) + ' |')
a7 = J('rq1/a7_sub.json')
w('\n### A7. Substantial modification (theta = share of the unit\'s PR-added lines modified), primary design\n\n| Unit, theta | Units | Mod. rate PR-weighted (Sup / Devin / Human) | Agent HR | Supervised HR | Devin HR |\n|---|---|---|---|---|---|')
for k, v in (a7 or {}).items():
    gg = v['groups']
    w(f"| {k}% | {v['n']:,} | {pc(gg['Supervised assistants']['mod_prw'])} / {pc(gg['Devin (autonomous)']['mod_prw'])} / {pc(gg['Human']['mod_prw'])} | {hr(v['within']['agent'])} | {hr(v['within']['supervised'])} | {hr(v['within']['devin'])} |")
m = J('rq1/a_misc.json') or {}
a8 = m.get('A8')
w('\n### A8. Attribution audit vs. the mined cohort\n')
if a8:
    w(f"Of the {a8['audited']} audited PRs (drawn from the 2,092 agent PRs with a located merge commit), **{a8['in_mined']} are in the mined cohort** (2,802 PRs), and {a8['in_primary']} in the primary design. The other {a8['audited'] - a8['in_mined']} PRs add no tracked source line.\n")
    w('| Scope | n | agent | mixed | human | unclear | agent or mixed |\n|---|---|---|---|---|---|---|')
    for k in ['mined|all', 'mined|Supervised assistants', 'mined|Devin (autonomous)']:
        x = a8[k]; f = lambda t: f'{t[0]} ({pc(t[1])} [{pc(t[2])}, {pc(t[3])}])'
        w(f"| {k} | {x['n']} | {f(x['agent'])} | {f(x['mixed'])} | {f(x['human'])} | {f(x['unclear'])} | {f(x['agent_or_mixed'])} |")
    w(f"\nLabels of mined audited PRs by agent: {a8['by_agent_mined']}. Intervals are 95% Wilson score intervals.\n")
    w('Primary RQ1 (line) under attribution restrictions:\n\n| Restriction | Units | PRs | Agent PRs | Agent HR | Supervised HR | Devin HR |\n|---|---|---|---|---|---|---|')
    for k in ['rq1_excl_human', 'rq1_excl_human_unclear', 'rq1_agent_label_only_vs_all_human', 'rq1_agent_or_mixed_label_vs_all_human']:
        x = a8.get(k)
        if x: w(f"| {k[4:]} | {x['n']:,} | {x['prs']:,} | {x['groups']['Agent (all)']['prs']} | {hr(x['within']['agent'])} | {hr(x['within']['supervised'])} | {hr(x['within']['devin'])} |")
a9 = (J('rq1/a9.json') or {}).get('A9')
w('\n### A9. File-level covariates at merge time (primary design)\n')
if a9:
    w('Covariates per unit\'s file, read from local clones by `src/mining/a9_extract.py` (shipped in `data/file_covariates/`): prior 3-month modification count (commits reachable from the PR base touching the path in the 90 days before the merge; the PR itself excluded), file age in days (first commit adding the path; 0 for new files), new file (path absent at the PR base), test path (directory test/tests/spec/specs/__tests__ or "test"/"spec" in the file name). Counts and age enter as log1p.\n')
    w('| Level | Units with features | Agent HR, same rows, base model | Agent HR + file covariates | Supervised base -> +cov | Devin base -> +cov |\n|---|---|---|---|---|---|')
    for g, x in a9.items():
        b, c = x['base_same_rows'], x['with_file_covariates']
        w(f"| {g} | {x['units_with_features']:,} / {x['units']:,} | {hr(b['agent'])} | {hr(c['agent'])} | {hr(b['supervised'])} -> {hr(c['supervised'])} | {hr(b['devin'])} -> {hr(c['devin'])} |")
    w('\nCovariate HRs (agent model, line level): ' + '; '.join(f'{k} {hr(v)}' for k, v in a9.get('line', {}).get('covariate_hrs', {}).items()))
    w('\nPR-weighted descriptives (line level): ' + '; '.join(f"{g}: new file {pc(d['new_file'])}, test path {pc(d['test'])}, median prior 3-month changes {d['prior_mod_3m_median']:.0f}, median file age {d['file_age_median']:.0f} d" for g, d in a9.get('line', {}).get('desc', {}).items()))
else: w('PENDING\n')
a10 = m.get('A10')
w('\n### A10. Repository-activity terciles (primary line design)\n\nCommit counts on the default branch are not stored for the cohort; the two activity measures are those of the Q1 filter: AIDev PR count of the repository (agent + human, all states, `data/repo_meta_v6.csv: total_pr_count`) and open issues. Terciles by rank.\n\n| Measure, tercile (range, repos) | Units | PRs | Agent HR | Supervised HR | Devin HR |\n|---|---|---|---|---|---|')
for k, x in (a10 or {}).items():
    w(f"| {k} ({x['range']['min']}-{x['range']['max']}, {x['range']['size']}) | {x['n']:,} | {x['prs']:,} | {hr(x['within']['agent'])} | {hr(x['within']['supervised'])} | {hr(x['within']['devin'])} |")
w('\nThe Q1-removed repositories themselves were never mined, so a "keep Q1" sensitivity is not possible without new mining.\n')
a11 = m.get('A11')
w('### A11. Descriptives per group\n')
if a11:
    w('| Group | All lines | Share of all lines in vendored paths | Non-vendored lines | Share in test paths (unweighted / PR-weighted) | PRs | Merge date Q1 / median / Q3 (min-max) |\n|---|---|---|---|---|---|---|')
    for g, x in a11.items():
        w(f"| {g} | {x['lines_all']:,} | {pc(x['share_vendored_all_lines'])} | {x['lines_nv']:,} | {pc(x['share_test_nv'])} / {pc(x['share_test_nv_prw'])} | {x['prs']} | {x['merged_q1']} / {x['merged_median']} / {x['merged_q3']} ({x['merged_min']} to {x['merged_max']}) |")
    w('\nMerged PRs per month: ' + '; '.join(f"{g}: {x['by_month']}" for g, x in a11.items() if g != 'Agent (all)'))
    w('\nTest paths here: directory test/tests/spec/__tests__ or "test" in the file name (A9 also counts "spec" in the name). New-file shares are in A9.\n')
w('### A12. Table 3 for the primary design\n\n| Level | Group | Units | PRs | % modified unweighted | % modified PR-weighted | Median days to modification (unweighted / PR-weighted) | Median follow-up of censored units (days) |\n|---|---|---|---|---|---|---|---|')
for g, r in R.items():
    for k, x in ((r or {}).get('A12') or {}).items():
        w(f"| {g} | {k} | {x['units']:,} | {x['prs']:,} | {pc(x['mod_unw'])} | {pc(x['mod_prw'])} | {x['median_days_to_mod']:.2f} / {x['median_days_to_mod_prw']:.2f} | {x['median_followup_censored']:.1f} |")
a13 = m.get('A13')
w('\n### A13. KM figure\n\n`figures/rq1_km_line_primary.pdf` (and .png): PR-weighted KM curves, primary line data, no title, legend colours match the lines, untruncated group names in the at-risk table.\n')
if a13:
    w('Lines at risk: ' + '; '.join(f'{g}: {v}' for g, v in a13['at_risk'].items()))
    for g, x in a13['info'].items():
        w(f"- {g}: last unit at risk at day {x['last_day_at_risk']:.1f}. Largest PR-weighted step at day {x['largest_step_day']:.1f} (drop {x['largest_step_size']:.4f}), all from PR {x['largest_step_top_pr']}. Largest unweighted step at day {x['unw_largest_step_day']:.1f} (drop {x['unw_largest_step_size']:.4f}), mostly PR {x['unw_step_top_pr']} ({x['unw_step_top_pr_lines']} lines).")
    w('\nFollow-up ends on 2025-12-31, so groups whose PRs were merged later (A11) have no unit at risk at the longest durations.\n')
pass
c = J('rq2/rq2_fb.json')
w('\n## C. RQ2 (primary units: non-vendored lines; word-boundary classifier; one observation per modifying commit x author group)\n')
if c:
    w(f"All lines (with vendored paths), word classifier (all lines, word, commit x group): n={c['repro_all_lines_word']['n']:,}, chi2={c['repro_all_lines_word']['chi2']:.2f}, V={c['repro_all_lines_word']['cramers_v']:.4f} .\n")
    for k, name in [('primary_word', 'Primary'), ('primary_word_no_merge', 'Excluding merge commits'), ('primary_word_single_group_commits', 'Only commits touching one group'),
                    ('primary_substring', 'Substring classifier'), ('file_commit_word', 'File x commit unit (non-vendored files), word'), ('file_commit_substring', 'File x commit unit, substring')]:
        x = c[k]
        w(f"**{name}**: n={x['n']:,} (agent {x['n_agent']:,}, human {x['n_human']:,}), chi2={x['chi2']:.2f}, dof={x['dof']}, p={x['p']:.3g}, Cramer's V={x['cramers_v']:.4f}\n")
        w('| Class | Agent % | Human % | Std. residual agent | Std. residual human |\n|---|---|---|---|---|')
        for cl in x['pct']:
            w(f"| {cl} | {x['pct'][cl]['Agent']:.2f} | {x['pct'][cl]['Human']:.2f} | {x['std_resid']['Agent'][cl]:+.2f} | {x['std_resid']['Human'][cl]:+.2f} |")
        w('')
    mc = c['merge_commits']
    w(f"Merge commits (more than one parent, from data/rq2/commit_parents.csv): {mc['distinct_commits']} distinct modifying commits, {mc['units']} of {mc['units_total']} units (by group {mc['merge_by_group']}; classified as {mc['merge_intent']}). Commits touching both groups: {c['commits_touching_both_groups']}. Word vs substring agreement over all commits: {pc(c['classifier_agreement_commits'])}.\n")
else: w('PENDING\n')
# D
def rq3a(d):
    s = f'{d}/results/rq31_bow_tournament_summary_combined_count.csv'
    if not os.path.exists(s): return 'PENDING (tournament running)\n'
    t = pd.read_csv(s).sort_values('auc_pr_mean', ascending=False)
    out = '| Model | AUC-PR mean [CI] | AUC-ROC mean [CI] | F1 | MCC |\n|---|---|---|---|---|\n' + '\n'.join(
        f"| {r.model} | {r.auc_pr_mean:.4f} [{r.auc_pr_ci_lower:.4f}, {r.auc_pr_ci_upper:.4f}] | {r.auc_roc_mean:.4f} [{r.auc_roc_ci_lower:.4f}, {r.auc_roc_ci_upper:.4f}] | {r.f1_mean:.4f} | {r.mcc_mean:.4f} |" for r in t.itertuples())
    for mtr in ['auc_pr', 'auc_roc']:
        p = f'{d}/results/rq3a_sk_{mtr}.csv'
        if os.path.exists(p): out += f'\n\nScott-Knott ESD ({mtr}): ' + ', '.join(f'{r.model} (rank {r.rank})' for r in pd.read_csv(p).itertuples())
    for nm, p in [('LR baseline (log file lines + log prior 3-month changes, same 30x10 grouped CV)', 'd1_lr_size_history.json'), ('Out-of-sample explanation', 'rq3a_explain_summary.json'),
                  ('Line localization', 'rq3a_line_localization_summary.json'), ('LIME runtime', 'lime_runtime.json')]:
        x = J(f'{d}/results/{p}')
        if x: out += f'\n\n{nm}: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k not in ('lime_top_tokens',)})
        else: out += f'\n\n{nm}: PENDING'
    return out + '\n'
def rq3b(d):
    s = f'{d}/results/rq32_process_tournament_summary.csv'
    if not os.path.exists(s): return 'PENDING (tournament running)\n'
    t = pd.read_csv(s); t = t[t.metric.isin(['macro_f1', 'auc_roc'])].pivot_table(index='model', columns='metric', values='mean').sort_values('macro_f1', ascending=False)
    out = '| Model | Macro F1 | AUC-ROC |\n|---|---|---|\n' + '\n'.join(f'| {k} | {r.macro_f1:.4f} | {r.auc_roc:.4f} |' for k, r in t.iterrows())
    for mtr in ['macro_f1', 'auc_roc']:
        p = f'{d}/results/rq3b_sk_{mtr}.csv'
        if os.path.exists(p): out += f'\n\nScott-Knott ESD ({mtr}): ' + ', '.join(f'{r.model} (rank {r.rank})' for r in pd.read_csv(p).itertuples())
    for nm, p in [('Out-of-sample explanation', 'rq3b_explain_summary.json'), ('LIME runtime', 'lime_runtime.json')]:
        x = J(f'{d}/results/{p}')
        if x: out += f'\n\n{nm}: ' + json.dumps(x)[:3000]
        else: out += f'\n\n{nm}: PENDING'
    return out + '\n'
w('## D. RQ3 (vendored/generated files excluded from all inputs)\n')
w('```\n' + (open(f'{H}/rq3/prep_log.txt').read() if os.path.exists(f'{H}/rq3/prep_log.txt') else 'see data/README.md\n') + '```\n')
w('### D1. RQ3a, agent files (full 8-model 30x10 tournament)\n'); w(rq3a(f'{H}/rq3/rq3a_agent'))
w('### D1. RQ3a, all files (comparison only: RandomForest, CatBoost, XGBoost + LR baseline, 30x10)\n'); w(rq3a(f'{H}/rq3/rq3a_all'))
w('### D2. RQ3b, PR-history leakage fix\n')
w('PR-history features (full leakage fix, `src/mining/extract_rq3_pr_features_fixed.py`): prior = author PRs created before T minus the PR itself; acceptance = share of those merged before T.\n')
w('```\n' + (open(f'{H}/rq3/prep_log_full_fix.txt').read() if os.path.exists(f'{H}/rq3/prep_log_full_fix.txt') else 'see data/README.md\n') + '```\n')
for sub in ['agent', 'all']:
    w(f'#### RQ3b full fix, {sub} files\n'); w(rq3b(f'{H}/rq3/rq3b_{sub}'))
pass
open(f'{H}/SUMMARY.md', 'w').write('\n'.join(L) + '\n'); print('SUMMARY.md written', len(L), 'lines')
