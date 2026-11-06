"""funnel_replay.py: filtering funnel of the earlier run, replaying filter_repos.py step by step offline with the cached
metadata and code-ratio files (REBUILD_DIR, default rebuild/). Diagnostic only: the code-ratio cache of the final run is
not available, so the last two rows differ from Table 1 (see src/analysis/funnel.py).
Usage (package root): python src/mining/funnel_replay.py <unused> data/cohort/prs_cohort256.csv"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import filter_repos as F
RB = os.environ.get("REBUILD_DIR", "rebuild")
a=pd.read_parquet(RB + '/aidev/all_pull_request.parquet'); h=pd.read_parquet(RB + '/aidev/human_pull_request.parquet')
a['slug']=a.repo_url.apply(F.normalize_slug); h['slug']=h.repo_url.apply(F.normalize_slug)
rows=[]
def log(step, slugs):
    sa=a[a.slug.isin(slugs)]; sh=h[h.slug.isin(slugs)]
    rows.append(dict(step=step, repos=len(slugs), agent_prs=len(sa), human_prs=len(sh),
        agent_merged=sa.merged_at.notna().sum(), human_merged=sh.merged_at.notna().sum()))
log('0 AIDev: all agent repos', set(a.slug))
cohort=set(a.slug)&set(h.slug); log('1 Cohort: repos with agent and human PRs', cohort)
m=pd.read_csv('data/cohort/repo_metadata_v1.csv'); m=m[m.slug.isin(cohort)].drop_duplicates('slug')
log('1b metadata available', set(m.slug))
m=m[m.stars>=F.MIN_STARS]; log('2 Stars >= 500', set(m.slug))
m=m[~m.license.apply(F.is_bad_license)]; log('3 Software license', set(m.slug))
m=m[m.is_archived==False]; log('4a Not archived', set(m.slug))
m=m[m.has_release==True]; log('4b Has release/tag', set(m.slug))
u=pd.concat([a[a.slug.isin(m.slug)][['slug','user']],h[h.slug.isin(m.slug)][['slug','user']]])
cc=u.groupby('slug').user.nunique(); m=m[m.slug.map(cc).fillna(0)>=2]; log('4c >= 2 contributors', set(m.slug))
m=m.copy(); m['total_pr_count']=m.slug.map(a.groupby('slug').size()).fillna(0)+m.slug.map(h.groupby('slug').size()).fillna(0)
out=[]
for lang,d in m.groupby('language'):
    if pd.isna(lang) or lang=='Unknown': continue
    if len(d)<4: out.append(d); continue
    out.append(d[(d.total_pr_count>d.total_pr_count.quantile(.25))&(d.open_issues>d.open_issues.quantile(.25))&(d.size_kb>d.size_kb.quantile(.25))])
q1_removed=set(m.slug)-set(pd.concat(out).slug)
m=pd.concat(out); log('5 Q1 filter per language', set(m.slug))
m[['slug','url','language']].to_csv(RB + '/q1_passing_repos.csv',index=False)
r=pd.read_csv('data/cohort/code_ratio_v1.csv'); m=m.merge(r[['url','code_ratio']],on='url',how='inner'); log('6a code ratio available', set(m.slug))
out=[]
for lang,d in m.groupby('language'):
    if len(d)<3: out.append(d); continue
    mu,sd=d.code_ratio.mean(),d.code_ratio.std(); out.append(d[(d.code_ratio>=mu-2.17*sd)&(d.code_ratio<=mu+2.17*sd)])
m=pd.concat(out); log('6b Code ratio within 97% CI', set(m.slug))
df=pd.DataFrame(rows); df.to_csv(RB + '/filtering_funnel_replay.csv',index=False); print(df.to_string(index=False))
pkg=set(pd.read_csv(sys.argv[2]).slug); print("final == package cohort:", set(m.slug)==pkg); print("only in replay:", sorted(set(m.slug)-pkg)); print("only in package:", sorted(pkg-set(m.slug)))
pd.Series(sorted(q1_removed)).to_csv(RB + '/q1_removed_repos.csv',index=False,header=['slug']); print('Q1-removed repos:', len(q1_removed))
