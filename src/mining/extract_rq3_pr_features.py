"""
extract_rq3_pr_features.py: RQ3b PR-history features with the definitions of the earlier version, counted over ALL pull requests
of the repository (GitHub search API) instead of the AIDev agent-PR table only (fix: the earlier version gave 0 prior PRs for 99% of
human-authored files because human PRs are not in that table).

For each PR with a modified file (T = merge time = birth of the file unit):
  contributor_prior_prs        PRs by the same author in the repo created before T
  contributor_acceptance_rate  share of those that were merged
  project_open_prs_at_birth    PRs in the repo created before T and not closed before T
Usage: python extract_rq3_pr_features.py <files.parquet> <prs_cohort.csv> <aidev_dir> <out.jsonl>
Resumable: cached queries are read back from <out.jsonl>.cache
"""
import sys, os, json, time, subprocess
import pandas as pd

files, cohort, aidev, out = sys.argv[1:5]
cache_path = out + '.cache'
cache = {}
if os.path.exists(cache_path):
    for l in open(cache_path):
        r = json.loads(l); cache[r['q']] = r['n']
cf = open(cache_path, 'a')
_last = [0.0]


# repositories renamed/transferred after AIDev was collected: search does not follow redirects (422), so query the
# current name (found with search/repositories). microsoft/hydralab is no longer public -> stays missing
RENAMED = {'mendableai/firecrawl': 'firecrawl/firecrawl', 'dotnet/aspire': 'microsoft/aspire', 'sam-goodwin/alchemy': 'alchemy-run/alchemy',
           'stephendev0/stikdebug': 'StikDebug/StikDebug', 'afadil/wealthfolio': 'wealthfolio/wealthfolio',
           'willccbb/verifiers': 'PrimeIntellect-ai/verifiers', 'block/goose': 'aaif-goose/goose', 'danny-avila/librechat': 'LibreChat-AI/LibreChat',
           'jlowin/fastmcp': 'PrefectHQ/fastmcp', 'tobikodata/sqlmesh': 'SQLMesh/sqlmesh', 'microsoft/qsharp': 'microsoft/qdk',
           'microsoft/teams-ai': 'microsoft/teams-sdk'}


def author_q(user):
    if user == 'Copilot':
        return 'app/copilot-swe-agent'
    if user.endswith('[bot]'):
        return 'app/' + user[:-5]
    return user


def count(q):
    if q in cache:
        return cache[q]
    for attempt in range(8):
        wait = 2.05 - (time.time() - _last[0])  # search API: 30 requests / minute
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        r = subprocess.run(['gh', 'api', '-X', 'GET', 'search/issues', '-f', f'q={q}', '-f', 'per_page=1', '--jq', '.total_count'],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            n = int(r.stdout.strip()); cache[q] = n
            cf.write(json.dumps(dict(q=q, n=n)) + '\n'); cf.flush()
            return n
        if '422' in r.stderr or 'Validation' in r.stderr:  # e.g. unknown author
            cache[q] = None; cf.write(json.dumps(dict(q=q, n=None)) + '\n'); cf.flush()
            return None
        time.sleep(min(60 * (attempt + 1), 300))
    return None


f = pd.read_parquet(files)
k = f[f.event == 1][['slug', 'pr_number', 'birth_date']].drop_duplicates(['slug', 'pr_number'])
c = pd.read_csv(cohort)
users = pd.concat([pd.read_parquet(os.path.join(aidev, x + '.parquet'))[['id', 'user']]
                   for x in ['all_pull_request', 'human_pull_request']]).drop_duplicates('id')
c = c.merge(users, left_on='pr_id', right_on='id', how='left')
k = k.merge(c[['slug', 'number', 'user']].rename(columns={'number': 'pr_number'}), on=['slug', 'pr_number'], how='left')
print('PRs', len(k), 'queries <=', 4 * len(k), flush=True)
rows = []
with open(out, 'w') as fo:
    for i, x in enumerate(k.itertuples()):
        T = pd.Timestamp(x.birth_date).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        base = f'repo:{RENAMED.get(x.slug.lower(), x.slug)} is:pr'
        prior = merged = None
        if isinstance(x.user, str):
            a = author_q(x.user)
            prior = count(f'{base} author:{a} created:<{T}')
            merged = count(f'{base} author:{a} is:merged created:<{T}') if prior else 0
        created = count(f'{base} created:<{T}')
        closed = count(f'{base} closed:<{T}')
        r = dict(slug=x.slug, pr_number=int(x.pr_number), contributor=x.user,
                 contributor_prior_prs=prior, contributor_acceptance_rate=(merged / prior) if prior else 0.0,
                 project_open_prs_at_birth=(created - closed) if created is not None and closed is not None else None)
        fo.write(json.dumps(r) + '\n'); fo.flush()
        if i % 50 == 0:
            print(i, r, flush=True)
print('done', flush=True)
