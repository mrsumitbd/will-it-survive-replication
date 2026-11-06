import sys, types, os, pandas as pd
R=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
RB=os.environ.get('REBUILD_DIR','rebuild')
for mod in ['requests','dotenv']:
    sys.modules.setdefault(mod, types.ModuleType(mod))
sys.modules['dotenv'].load_dotenv = lambda *a, **k: None
sys.path.insert(0, R+'/src/mining'); import filter_repos as F
os.chdir(R)
a=pd.read_parquet(RB+'/aidev/all_pull_request.parquet'); h=pd.read_parquet(RB+'/aidev/human_pull_request.parquet')
a['slug']=a.repo_url.apply(F.normalize_slug); h['slug']=h.repo_url.apply(F.normalize_slug)
cohort=set(a.slug)&set(h.slug)
m=pd.read_csv('data/cohort/repo_metadata_v1.csv'); m=m[m.slug.isin(cohort)].drop_duplicates('slug')
m=m[m.stars>=F.MIN_STARS]; m=m[~m.license.apply(F.is_bad_license)]; m=m[m.is_archived==False]; m=m[m.has_release==True]
u=pd.concat([a[a.slug.isin(m.slug)][['slug','user']],h[h.slug.isin(m.slug)][['slug','user']]])
cc=u.groupby('slug').user.nunique(); m=m[m.slug.map(cc).fillna(0)>=2]
m=m.copy(); m['total_pr_count']=m.slug.map(a.groupby('slug').size()).fillna(0)+m.slug.map(h.groupby('slug').size()).fillna(0)
out=[]
for lang,d in m.groupby('language'):
    if pd.isna(lang) or lang=='Unknown': continue
    if len(d)<4: out.append(d); continue
    out.append(d[(d.total_pr_count>d.total_pr_count.quantile(.25))&(d.open_issues>d.open_issues.quantile(.25))&(d.size_kb>d.size_kb.quantile(.25))])
q=pd.concat(out); print('Q1 repos', len(q))
c=pd.read_csv('data/cohort/prs_cohort256.csv'); s246=set(c.slug)
print('246 subset of Q1:', s246<=set(q.slug), len(s246-set(q.slug)))
rest=q[~q.slug.isin(s246)][['slug','language','size_kb']].copy()
for nm,t in [('agent',a),('human',h)]:
    rest[nm+'_prs']=rest.slug.map(t.groupby('slug').size()).fillna(0).astype(int)
    rest[nm+'_merged']=rest.slug.map(t[t.merged_at.notna()].groupby('slug').size()).fillna(0).astype(int)
print(rest.sort_values(['agent_merged','human_merged']).to_string(index=False))
rest.to_csv(RB+'/q1_not_in_246.csv', index=False)
print('merged in 246: agent', a[a.slug.isin(s246)&a.merged_at.notna()].shape[0], 'human', h[h.slug.isin(s246)&h.merged_at.notna()].shape[0])
for s in ['elizaos/eliza','dust-tt/dust']:
    print(s, 'size_kb', m.set_index('slug').size_kb.get(s), 'merged agent', (c[(c.slug==s)].author_type=='Agent').sum(), 'human', (c[(c.slug==s)].author_type=='Human').sum())
print('size_kb quantiles in 246', m[m.slug.isin(s246)].size_kb.describe().to_dict())
