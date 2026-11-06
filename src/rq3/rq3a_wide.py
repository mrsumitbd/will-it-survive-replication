"""rq3a_wide.py: RQ3a tournament scores -> wide format (one row per repetition x fold, one column per model) for Scott-Knott ESD.
Usage: python rq3a_wide.py <scores.csv> <out_dir>"""
import sys, os
import pandas as pd
s = pd.read_csv(sys.argv[1])
for m in ['auc_pr', 'auc_roc', 'mcc', 'f1']:
    s.pivot_table(index=['repetition', 'fold'], columns='model', values=m).reset_index(drop=True).to_csv(
        os.path.join(sys.argv[2], f'rq3a_wide_{m}.csv'), index=False)
