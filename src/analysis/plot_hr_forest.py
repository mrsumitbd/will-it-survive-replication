"""plot_hr_forest.py: Figure 2 of the paper.

Forest plot of the within-repository line-level hazard ratios (agent-attributed vs. human code)
across the design choices reported in RQ1. Reads results/rq1/ (or the directory given as the first argument):
  rq1_line.json          A1_primary (PR-weighted), A1_nv_unw (unweighted)
  a4_ph_line.json        time_split (days 0-14, 14-90, 90+)
  c1_calendar_line.json  a_all_after_cut (merged on/after 2025-05-16), b_repo_x_month (strata)
  a14_tasktype.json      primary_slug_x_task (strata: repository x AIDev task type)
Writes <results>/figures/rq1_hr_forest.pdf and .png
"""
import os, sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS
BASE = sys.argv[1] if len(sys.argv) > 1 else RESULTS
O = os.path.join(BASE, 'rq1'); H = os.path.join(BASE, 'figures'); os.makedirs(H, exist_ok=True)
L = lambda f: json.load(open(os.path.join(O, f)))
line, ph, cal, tk = L('rq1_line.json'), L('a4_ph_line.json'), L('c1_calendar_line.json'), L('a14_tasktype.json')
ts = ph['time_split']

def tri(a, s, d):
    return {'agent': a, 'supervised': s, 'devin': d}

rows = [  # (group header, row label, {tier: {hr, lo, hi, p}})
    ('Weighting', 'PR-weighted (primary)', line['A1_primary']['within']),
    ('Weighting', 'Unweighted', line['A1_nv_unw']['within']),
    ('Follow-up period', 'Days 0 to 14', tri(ts['agent_0_14']['is_agent'], ts['tier_0_14']['supervised'], ts['tier_0_14']['devin'])),
    ('Follow-up period', 'Days 14 to 90', tri(ts['agent_14_90']['is_agent'], ts['tier_14_90']['supervised'], ts['tier_14_90']['devin'])),
    ('Follow-up period', 'After day 90', tri(ts['agent_90_9999']['is_agent'], ts['tier_90_9999']['supervised'], ts['tier_90_9999']['devin'])),
    ('Merge date', 'Merged after mid-May 2025', cal['a_all_after_cut']['within']),
    ('Merge date', 'Repository $\\times$ merge month', tri(cal['b_repo_x_month']['is_agent'], cal['b_repo_x_month']['supervised'], cal['b_repo_x_month']['devin'])),
    ('Task type', 'Repository $\\times$ task type', tk['primary_slug_x_task']),
]
tiers = [('agent', 'All agents', '#000000', 'o'),
         ('supervised', 'Supervised assistants', '#d62728', 's'),
         ('devin', 'Devin', '#2ca02c', '^')]

plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.0, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.4, 3.35))
y, yt, yl, heads, prev = 0.0, [], [], [], None
for head, lab, r in rows:
    if head != prev:
        y -= 0.55 if prev else 0.0
        heads.append((y, head)); y -= 0.9; prev = head
    for k, (t, _, c, m) in enumerate(tiers):
        v = r[t]; yy = y + (1 - k) * 0.24
        ax.plot([v['lo'], v['hi']], [yy, yy], color=c, lw=0.9, solid_capstyle='butt')
        ax.plot(v['hr'], yy, marker=m, ms=3.6, color=c, mec=c, mew=0.8,
                mfc=c if v['p'] < 0.05 else 'white', ls='none')
    yt.append(y); yl.append(lab); y -= 1.0
ax.axvline(1.0, color='0.5', lw=0.6, ls='--', zorder=0)
ax.set_xscale('log'); ax.set_xlim(0.33, 1.8)
ax.xaxis.set_major_locator(FixedLocator([0.5, 0.75, 1.0, 1.5]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(['0.5', '0.75', '1', '1.5'])
ax.set_yticks(yt); ax.set_yticklabels(yl)
for yy, h in heads:
    ax.text(-0.02, yy, h, transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold')
ax.set_ylim(y + 0.4, heads[0][0] + 0.6)
ax.set_xlabel('Hazard ratio vs. human code (log scale, 95% CI)')
ax.text(0.345, heads[0][0] + 0.25, 'modified later', ha='left', va='bottom', fontsize=6, color='0.35')
ax.text(1.75, heads[0][0] + 0.25, 'modified sooner', ha='right', va='bottom', fontsize=6, color='0.35')
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
hs = [plt.Line2D([], [], color=c, marker=m, ms=3.6, lw=0.9, label=n) for _, n, c, m in tiers]
ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.35, -0.16), ncol=3, frameon=False,
          handlelength=1.4, columnspacing=1.0, fontsize=6.5)
fig.savefig(os.path.join(H, 'rq1_hr_forest.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(H, 'rq1_hr_forest.png'), dpi=200, bbox_inches='tight')
for head, lab, r in rows:
    print(f'{lab:28s}', '  '.join(f"{t}={r[t]['hr']:.2f} [{r[t]['lo']:.2f}, {r[t]['hi']:.2f}] p={r[t]['p']:.3f}" for t, *_ in tiers))
