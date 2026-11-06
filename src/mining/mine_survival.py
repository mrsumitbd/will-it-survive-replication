"""
mine_survival.py  (revision of the earlier line/file mining)

For every merged PR in a repository:
  * birth  = the PR's merge commit on the default branch (found by GitHub merge time
             matched against first-parent committer dates, message "#N" as tie-break)
  * units  = every added source line in the PR's net diff (no sampling caps)
  * line death = the first commit, after the last commit in which the line still exists,
             that touches the file (git blame --reverse M..C), where C is the last
             default-branch commit on or before the cutoff (Dec 31 2025)
  * file death = first commit after M (up to C) that touches the file (paper's definition)
  * hunk  = contiguous run of added lines in one diff hunk; derived from its lines later

Usage: python mine_survival.py <slug> <prs.csv> <out_dir> [--blame-opts "-w -M -C"]
prs.csv columns: slug, number, merged_at, author_type, agent_name
"""
import argparse, os, re, shutil, subprocess, sys, json, tempfile, time
from datetime import datetime, timezone, timedelta
import pandas as pd

CUTOFF = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
SOURCE_EXT = ('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.h', '.hpp',
              '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt')
MATCH_WINDOW_S = 180


def git(repo, *args, check=True):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True, errors='replace')
    if check and r.returncode != 0:
        raise RuntimeError(f'git {" ".join(args[:3])}: {r.stderr[:300]}')
    return r.stdout


def parse_dt(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)


def first_parent_log(repo):
    out = git(repo, 'log', '--first-parent', '--format=%H%x1f%cI%x1f%P%x1f%s%x00', 'HEAD')
    rows = []
    for line in out.split('\x00'):
        line = line.strip('\n')
        if not line:
            continue
        h, cd, parents, subj = line.split('\x1f', 3)
        rows.append((h, parse_dt(cd), parents.split(), subj))
    return rows  # newest first


def find_merge(fp, pr_number, merged_at):
    """Return (first_commit, last_commit, method). Range covers rebase-merges."""
    near = [r for r in fp if abs((r[1] - merged_at).total_seconds()) <= MATCH_WINDOW_S]
    tag = re.compile(rf'#{pr_number}(?!\d)')
    if near:
        tagged = [r for r in near if tag.search(r[3])]
        cand = tagged or near
        if len(cand) > 1 and not tagged:
            # several untagged commits at the merge time: accept only if they are one
            # consecutive first-parent run (a rebase-merge); range = oldest..newest
            idxs = sorted(fp.index(r) for r in cand)
            if idxs == list(range(idxs[0], idxs[-1] + 1)) and all(len(fp[i][2]) == 1 for i in idxs):
                return fp[idxs[-1]][0], fp[idxs[0]][0], 'time_rebase_run'
            return None, None, 'ambiguous_time'
        # rebase-merge: consecutive first-parent commits sharing the merge time
        idx = fp.index(cand[0])
        j = idx
        while j + 1 < len(fp) and abs((fp[j + 1][1] - fp[idx][1]).total_seconds()) <= 2 \
                and not tag.search(fp[j + 1][3]) and len(fp[j + 1][2]) == 1 and not tagged:
            j += 1
        return fp[j][0], fp[idx][0], 'time+msg' if tagged else 'time'
    tagged = [r for r in fp if tag.search(r[3]) and abs((r[1] - merged_at).total_seconds()) <= 86400 * 3]
    if len(tagged) == 1:
        return tagged[0][0], tagged[0][0], 'msg_within_3d'
    return None, None, 'not_found'


HUNK_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


def added_lines(repo, base, tip):
    """{path: [(line_no, hunk_idx, text)]} for added lines in base..tip (net diff)."""
    out = git(repo, 'diff', '--unified=0', '--no-color', '-M', base, tip)
    res, path, hunk_idx = {}, None, -1
    for line in out.split('\n'):
        if line.startswith('+++ '):
            p = line[4:]
            path = None if p == '/dev/null' else p[2:] if p.startswith('b/') else p
            hunk_idx = -1
            if path and not path.lower().endswith(SOURCE_EXT):
                path = None
        elif line.startswith('@@') and path:
            m = HUNK_RE.match(line)
            cur = int(m.group(1)); hunk_idx += 1
        elif path and line.startswith('+') and not line.startswith('+++'):
            res.setdefault(path, []).append((cur, hunk_idx, line[1:]))
            cur += 1
    return res


def blame_reverse(repo, tip, cutoff_sha, path, linenos, opts):
    """{line_no: last_sha_where_line_exists}"""
    ranges = []
    for n in sorted(linenos):
        if ranges and n == ranges[-1][1] + 1:
            ranges[-1][1] = n
        else:
            ranges.append([n, n])
    args = ['blame', '--reverse', '--porcelain', *opts]
    for a, b in ranges:
        args += ['-L', f'{a},{b}']
    args += [f'{tip}..{cutoff_sha}', '--', path]
    out = git(repo, *args, check=False)
    res = {}
    for line in out.split('\n'):
        m = re.match(r'^([0-9a-f]{40}) \d+ (\d+)', line)
        if m:
            res[int(m.group(2))] = m.group(1)
    return res


def death_commit(repo, last_sha, cutoff_sha, path, cache):
    """First commit after last_sha (towards cutoff) that touches path; prefer earliest date."""
    key = (last_sha, path)
    if key in cache:
        return cache[key]
    out = git(repo, 'log', '--ancestry-path', '--format=%H%x1f%cI%x1f%P', f'{last_sha}..{cutoff_sha}', '--', path, check=False)
    best = None
    for line in out.split('\n'):
        if not line:
            continue
        h, cd, parents = line.split('\x1f')
        if last_sha in parents.split():
            d = parse_dt(cd)
            if best is None or d < best[1]:
                best = (h, d)
    if best is None:  # fall back: any commit in range touching path, earliest
        for line in out.split('\n'):
            if not line:
                continue
            h, cd, _ = line.split('\x1f'); d = parse_dt(cd)
            if best is None or d < best[1]:
                best = (h, d)
    if best is None:  # path-limited log can miss renames/deletions: take children of last_sha
        out = git(repo, 'rev-list', '--ancestry-path', '--parents', f'{last_sha}..{cutoff_sha}', check=False)
        for line in out.split('\n'):
            parts = line.split()
            if len(parts) > 1 and last_sha in parts[1:]:
                cd = git(repo, 'log', '-1', '--format=%cI', parts[0], check=False).strip()
                if cd:
                    d = parse_dt(cd)
                    if best is None or d < best[1]:
                        best = (parts[0], d)
    cache[key] = best
    return best


def file_death(repo, tip, cutoff_sha, path):
    out = git(repo, 'log', '--first-parent', '--reverse', '--format=%H%x1f%cI', f'{tip}..{cutoff_sha}', '--', path, check=False)
    lines = [x for x in out.split('\n') if x]
    if not lines:
        return None
    h, cd = lines[0].split('\x1f')
    return h, parse_dt(cd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slug'); ap.add_argument('prs'); ap.add_argument('out')
    ap.add_argument('--blame-opts', default='')
    ap.add_argument('--keep-clone', action='store_true')
    ap.add_argument('--branch', default=None, help='branch to mine when the current default branch history was rewritten')
    ap.add_argument('--clone-dir', default=None)
    a = ap.parse_args()
    opts = a.blame_opts.split()
    os.makedirs(a.out, exist_ok=True)
    tag = a.slug.replace('/', '__')
    done = os.path.join(a.out, f'{tag}.done.json')
    if os.path.exists(done):
        return
    prs = pd.read_csv(a.prs)
    prs = prs[prs.slug == a.slug]
    repo = a.clone_dir or tempfile.mkdtemp(prefix='mine_')
    t0 = time.time()
    meta = {'slug': a.slug, 'n_prs': len(prs), 'blame_opts': a.blame_opts}
    try:
        if not os.path.exists(os.path.join(repo, '.git')):
            br = ['--branch', a.branch] if a.branch else []
            meta['branch'] = a.branch or 'default'
            r = subprocess.run(['git', 'clone', '--quiet', '--single-branch', *br, f'https://github.com/{a.slug}.git', repo],
                               capture_output=True, text=True)
            if r.returncode != 0:
                meta['error'] = 'clone_failed: ' + r.stderr[:200]
                json.dump(meta, open(done, 'w')); return
        fp = first_parent_log(repo)
        cut = [r for r in fp if r[1] <= CUTOFF]
        if not cut:
            meta['error'] = 'no_commit_before_cutoff'; json.dump(meta, open(done, 'w')); return
        cutoff_sha = cut[0][0]
        meta['cutoff_sha'] = cutoff_sha
        line_rows, file_rows, pr_rows, dcache, commit_info = [], [], [], {}, {}
        for _, pr in prs.iterrows():
            merged_at = parse_dt(str(pr.merged_at))
            rec = dict(slug=a.slug, pr_number=int(pr.number), author_type=pr.author_type, agent_name=pr.agent_name,
                       merged_at=merged_at.isoformat())
            if merged_at > CUTOFF:
                rec['status'] = 'after_cutoff'; pr_rows.append(rec); continue
            first, last, method = find_merge(fp, int(pr.number), merged_at)
            rec['match_method'] = method
            if not first:
                rec['status'] = 'merge_not_found'; pr_rows.append(rec); continue
            parents = git(repo, 'rev-list', '--parents', '-n', '1', first).split()
            base = parents[1] if len(parents) > 1 else None
            if base is None:
                rec['status'] = 'root_commit'; pr_rows.append(rec); continue
            birth_dt = [r[1] for r in fp if r[0] == last][0]
            rec.update(birth_sha=last, base_sha=base, status='ok')
            adds = added_lines(repo, base, last)
            rec['n_source_files'] = len(adds); rec['n_added_lines'] = sum(len(v) for v in adds.values())
            pr_rows.append(rec)
            for path, lines in adds.items():
                fd = file_death(repo, last, cutoff_sha, path)
                file_rows.append(dict(slug=a.slug, pr_number=int(pr.number), unit_id=path, birth_sha=last,
                                      birth_date=birth_dt.isoformat(), event=int(fd is not None),
                                      death_sha=fd[0] if fd else None, death_date=fd[1].isoformat() if fd else None,
                                      n_added_lines=len(lines)))
                last_seen = blame_reverse(repo, last, cutoff_sha, path, [l[0] for l in lines], opts)
                for (n, h_idx, text) in lines:
                    ls = last_seen.get(n)
                    if ls is None:
                        ev, dsha, ddt, kind = None, None, None, 'blame_failed'
                    elif ls == cutoff_sha:
                        ev, dsha, ddt, kind = 0, None, None, 'censored'
                    else:
                        dc = death_commit(repo, ls, cutoff_sha, path, dcache)
                        if dc is None:
                            ev, dsha, ddt, kind = None, None, None, 'death_commit_not_found'
                        else:
                            ev, dsha, ddt, kind = 1, dc[0], dc[1], 'died'
                    line_rows.append(dict(slug=a.slug, pr_number=int(pr.number), file=path, line=n, hunk=h_idx,
                                          birth_sha=last, birth_date=birth_dt.isoformat(), event=ev, kind=kind,
                                          last_seen_sha=ls, death_sha=dsha,
                                          death_date=ddt.isoformat() if ddt else None,
                                          blank=int(text.strip() == '')))
        # commit messages for all death commits (RQ2)
        dshas = {r['death_sha'] for r in line_rows if r['death_sha']} | {r['death_sha'] for r in file_rows if r['death_sha']}
        for h in dshas:
            msg = git(repo, 'log', '-1', '--format=%an%x1f%ae%x1f%B', h, check=False)
            parts = msg.split('\x1f', 2)
            commit_info[h] = dict(sha=h, author=parts[0] if parts else '', email=parts[1] if len(parts) > 1 else '',
                                  message=parts[2].strip() if len(parts) > 2 else '')
        pd.DataFrame(line_rows).to_parquet(os.path.join(a.out, f'{tag}.lines.parquet'))
        pd.DataFrame(file_rows).to_parquet(os.path.join(a.out, f'{tag}.files.parquet'))
        pd.DataFrame(pr_rows).to_parquet(os.path.join(a.out, f'{tag}.prs.parquet'))
        pd.DataFrame(list(commit_info.values())).to_parquet(os.path.join(a.out, f'{tag}.commits.parquet'))
        meta.update(n_lines=len(line_rows), n_files=len(file_rows), seconds=round(time.time() - t0, 1))
    except Exception as e:
        meta['error'] = repr(e)[:500]
    finally:
        if not a.keep_clone and not a.clone_dir:
            shutil.rmtree(repo, ignore_errors=True)
    json.dump(meta, open(done, 'w'))


if __name__ == '__main__':
    main()
