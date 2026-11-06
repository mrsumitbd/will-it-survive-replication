"""Code ratio (LOC/(LOC+CLOC)) as in filter_repos.analyze_code_ratio, but without the 120 s clone / 300 s cloc timeouts."""
import sys, json, subprocess, tempfile, shutil, csv
url = sys.argv[1]; out = sys.argv[2]
d = tempfile.mkdtemp()
try:
    r = subprocess.run(['git', 'clone', '-q', '--depth', '1', url, d], capture_output=True, timeout=3600)
    res = {'url': url, 'error': 'clone_failed'} if r.returncode else None
    if res is None:
        c = subprocess.run(['cloc', '--json', '--quiet', d], capture_output=True, text=True, timeout=3600)
        s = json.loads(c.stdout).get('SUM', {})
        code, com = s.get('code', 0), s.get('comment', 0)
        res = {'url': url, 'code_ratio': code / (code + com) if code + com else 0.0, 'loc': code, 'comments': com,
               'total_lines': code + com + s.get('blank', 0)}
except Exception as e:
    res = {'url': url, 'error': repr(e)[:200]}
finally:
    shutil.rmtree(d, ignore_errors=True)
with open(out, 'a') as f:
    f.write(json.dumps(res) + '\n')
