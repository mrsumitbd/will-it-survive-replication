"""sanitize_commits.py: remove personal data from the commit table before it is shipped.

Drops the author name and e-mail columns of commits.parquet and replaces every personal e-mail address inside the
commit messages (for example in Co-authored-by trailers) with "<email>". The RQ2 classifiers only read the messages; the
script checks that the intent labels of every commit (whole-word and substring classifiers) are unchanged.
Usage: python src/mining/sanitize_commits.py <in commits.parquet> <out commits.parquet>
"""
import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis', 'ease_src'))
from mine_death_details import CLASSIFICATION_RULES, classify_message

EMAIL = re.compile(r'[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
# generic service addresses that are not personal data are kept (e.g. the GitHub signature support@github.com, whose
# word "support" is a keyword of the classifier)
KEEP = {'support@github.com', 'noreply@github.com', 'noreply@anthropic.com'}


def mask(s):
    return EMAIL.sub(lambda m: m.group() if m.group().lower() in KEEP else '<email>', s)
PRIORITY = ['Security', 'Corrective (Bug Fix)', 'Performance', 'Adaptive (New Feature)', 'Perfective (Refactor)',
            'Preventive (Testing)', 'Documentation', 'Chore/Maintenance']
WORD_RE = {c: [re.compile(r'(?<![a-z0-9])' + re.escape(k) + r'(?![a-z0-9])') for k in ks] for c, ks in CLASSIFICATION_RULES.items()}


def classify_word(msg):  # same rule as src/analysis/rq2_fb.py
    m = str(msg).lower(); hits = {c for c, regs in WORD_RE.items() if any(r.search(m) for r in regs)}
    return next((c for c in PRIORITY if c in hits), 'Other')


src, dst = sys.argv[1:3]
c = pd.read_parquet(src)
m = c.message.fillna('').map(mask)
same_word = (c.message.map(classify_word) == m.map(classify_word)).all()
same_sub = (c.message.map(lambda s: classify_message(s)[0]) == m.map(lambda s: classify_message(s)[0])).all()
print('messages changed:', int((m != c.message.fillna('')).sum()), '| labels unchanged (word, substring):', same_word, same_sub)
if not (same_word and same_sub):
    sys.exit('masking changes an intent label; not written')
out = c.drop(columns=[x for x in ['author', 'email'] if x in c.columns]).assign(message=m.where(c.message.notna(), None))
out.to_parquet(dst, index=False)
print('written', dst, list(out.columns), len(out))
