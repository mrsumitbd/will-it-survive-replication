"""Run rq3_explain.py with LIME calls timed (per explanation). Usage: python lime_timed.py rq3a|rq3b <scripts_dir>"""
import sys, time, json, runpy
mode, scripts = sys.argv[1], sys.argv[2]; sys.path.insert(0, scripts)
times = []
if mode == 'rq3a':
    import rq3a_tournament as T; f = T.apply_lime_explanations
    def g(*a, **k):
        t = time.time(); r = f(*a, **k); times.append((time.time() - t, len(r))); return r
    T.apply_lime_explanations = g
else:
    import lime_analysis_rq32_binned_process as L; f = L.run_lime_analysis
    def g(*a, **k):
        t = time.time(); r = f(*a, **k); times.append((time.time() - t, len(r[0]) if isinstance(r, tuple) else len(r))); return r
    L.run_lime_analysis = g
sys.argv = ['rq3_explain.py', mode, scripts]
runpy.run_path(f'{scripts}/rq3_explain.py', run_name='__main__')
tot = sum(t for t, _ in times); n = sum(k for _, k in times)
json.dump(dict(calls=len(times), explanations=n, seconds_total=tot, seconds_per_explanation=tot / n if n else None),
          open('results/lime_runtime.json', 'w'), indent=1)
print('LIME runtime', tot, 's for', n, 'explanations')
