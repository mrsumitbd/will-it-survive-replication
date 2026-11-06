import sys, types, os, pandas as pd
R=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root
RB=os.environ.get('REBUILD_DIR','rebuild')
for mod in ['requests','dotenv']: sys.modules.setdefault(mod, types.ModuleType(mod))
sys.modules['dotenv'].load_dotenv = lambda *a, **k: None
sys.path.insert(0, R+'/src/mining'); import filter_repos as F; os.chdir(R)
a=pd.read_parquet(RB+'/aidev/all_pull_request.parquet'); h=pd.read_parquet(RB+'/aidev/human_pull_request.parquet')
a['slug']=a.repo_url.apply(F.normalize_slug); h['slug']=h.repo_url.apply(F.normalize_slug)
s246=set(pd.read_csv('data/cohort/prs_cohort256.csv').slug)
nm=pd.read_csv(RB+'/q1_not_in_246.csv'); nm=nm[(nm.agent_merged==0)&(nm.human_merged==0)]
s256=s246|set(nm.slug); print('repos', len(s256))
for nmx,t in [('agent',a),('human',h)]:
    x=t[t.slug.isin(s256)]; print(nmx,'all PRs',len(x),'merged',x.merged_at.notna().sum())
print('human table rows',len(h),'distinct repo_url',h.repo_url.nunique())
