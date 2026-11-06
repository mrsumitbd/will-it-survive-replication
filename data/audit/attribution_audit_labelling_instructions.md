# Attribution audit: labelling instructions

Nov 2, 2026

## Purpose

The audit estimates how often the code in an "agent-attributed" PR was actually written by the agent. The answer bounds the measurement error in RQ1 to RQ3, where agent vs human is the independent variable.

- Reviewer 3 (EASE 2026) argued that AIDev's PR-level signals (bot accounts, branch prefixes, "Co-Authored-By" strings) are retrieval heuristics, not proof of authorship.
- The audit answers that point for the agent side. The human side is handled in the threats section: unmarked AI assistance cannot be checked from GitHub data.
- The unit is one merged PR from the study cohort. The question is about the code the PR adds, not about who opened it.

## Sample

The sheet holds 325 PRs, drawn at random from 2,092. The 2,092 are the agent-attributed merged PRs in the 256-repo cohort whose code units were mined (`status = ok`).

- **Size:** 95% confidence, ±5% margin, worst case p = 0.5, following Cochran's sample-size formula for a proportion with the finite-population correction (Cochran, *Sampling Techniques*, 1977). n0 = 1.96² × 0.25 / 0.05² = 384.2, and n = n0 / (1 + (n0 − 1) / N) with N = 2,092 gives 325 (rounded up).
- **Allocation:** stratified by agent with proportional allocation (Cochran 1977), which keeps the overall estimate self-weighting. Largest-remainder rounding, seed 42.
- **Order:** rows are shuffled (seed 42), so labelling in sheet order does not batch one agent.
- **Scope of the margin:** it covers the overall agent-authorship rate only. Per-agent rates for Cursor (11) and Claude Code (7) are too small to report with that margin.

| Agent | PRs in population | PRs in sample |
| --- | --- | --- |
| Copilot | 710 | 110 |
| OpenAI Codex | 670 | 104 |
| Devin | 597 | 93 |
| Cursor | 68 | 11 |
| Claude Code | 47 | 7 |
| **Total** | **2,092** | **325** |

Files: `ease-revision/attribution_audit_sheet.xlsx` (label dropdown, guide tab) and the same rows as `attribution_audit_sheet.csv`, in the project files.

## Labels

Pick exactly one label per PR, judged on the commits that add or change code (source files, not docs or lock files).

| Label | Use when |
| --- | --- |
| agent | Every substantive code change in the PR comes from the agent |
| mixed | The agent made substantive code changes, and a human also made substantive code changes in the PR |
| human | All substantive code changes come from a human; the agent's part was non-substantive or limited to the PR description or review |
| unclear | The commit history cannot be inspected (squashed with no agent marker, deleted branch, 404) or its signals contradict each other |

The split between agent and human contributions follows the mixed-authorship view of git history: a change is credited to the identity that authored the commit (Bird et al. 2011 use the same commit-level credit to measure ownership). The PR-level agent signals are the ones AIDev used to collect the PRs (Procedure).

**Decision rule for "substantive":** a code change is non-substantive only if it falls in one of the non-essential change categories of Kawrykow and Robillard (ICSE 2011): whitespace and documentation-related updates, trivial type updates, local variable extractions, rename-induced modifications (the reference updates a rename causes; the rename itself counts as essential), trivial keyword modifications (such as redundant `this`), and local variable renames. Merge commits from the base branch carry no new code and are ignored. Every other change to source code is substantive, however few lines it touches. There is no line-count threshold, because Purushothaman and Perry (TSE 2005) show that one-line changes carry real behaviour and defects, so size does not separate trivial from meaningful edits.

## Procedure

Work in sheet order (it is shuffled) and do not look at any RQ results while labelling.

1. Open `html_url`. Note who opened the PR and the head branch name.
2. Open the **Commits** tab. For every commit that changes source files, record the author, the committer and any `Co-authored-by` trailer (click the commit to see the full message).
3. Separate the agent's commits from human commits using the agent markers below. Ignore merge commits from the base branch.
4. For each human-authored code commit, decide whether it is substantive (Labels section).
5. Pick the label. In `evidence`, write the deciding fact in a few words (e.g. "5 commits by copilot-swe-agent; 1 human typo fix").
6. Use `notes` for anything odd: force-pushed history, squash merge, PR re-opened by a human.

PR-level agent signals, as AIDev used them to collect each agent's PRs (Li, Zhang and Hassan 2025, Table 2). These say why a PR is in the dataset; the commits decide the label.

| Agent | AIDev signal (GitHub search) | What to confirm in the commits |
| --- | --- | --- |
| OpenAI Codex | head branch starts with `codex/` | Which commits were created on that branch before any later pushes, and who pushed later commits |
| Devin | PR author `devin-ai-integration[bot]` | Which commits come from the Devin identity and which from people |
| GitHub Copilot | head branch starts with `copilot/` and PR user login `Copilot` | Which commits come from the Copilot identity and which from people |
| Cursor | head branch starts with `cursor/` | Which commits were made on that branch by the agent and which by people |
| Claude Code | PR body contains `Co-Authored-By: Claude` | Which commits carry that trailer; commits without it are human |

AIDev's signals are PR-level, and for Codex, Cursor and Claude Code the commit author shown by git can be the human who runs the agent. Record what the PR page shows rather than assuming a convention, and use `unclear` when the commits cannot tell the agent's changes apart from the human's.

## Edge cases

| Situation | Label |
| --- | --- |
| Squash-merged PR; the Commits tab still shows the agent's original commits | Judge the original commits |
| Squash-merged PR; only one human-authored commit is visible, no trailer, no agent branch | unclear |
| Human commits containing only non-essential changes (Labels section), such as whitespace, comment or doc fixes, local variable renames, or the reference updates a rename causes | agent |
| Human added tests or a new function after the agent's commits | mixed |
| Agent only fixed CI or lint on human code | human |
| PR opened by a human, but every code commit is by the agent bot | agent |
| Agent marker only in the PR body; commits are plain human commits (not Codex or Claude Code) | human, note "marker in body only" |
| PR page or repository no longer exists | unclear, note "404" |

When two rules seem to apply, the one about the code-adding commits wins over the one about who opened the PR.

## Recording and reporting

Fill `label`, `evidence` and `notes` in the xlsx; leave every other column unchanged so rows can be joined back by `sample_id`.

- **Single coder (decided 2026-11-02):** the first author labels all 325 PRs; there is no second rater. The ACM SIGSOFT Empirical Standards accept a single rater when the paper gives a reasonable justification, such as ratings that are uncontroversial or more raters being impractical (Ralph et al. 2021). Both apply: the labels rest on observable PR evidence (commit authors, bot identities, trailers, branch names) that needs little interpretation (McDonald, Schoenebeck and Forte 2019), and the team has no second annotator. Recent SE studies label with one annotator in the same way (Rabbi et al. 2026, a study of AI-generated PRs).
- **Mitigation:** follow this written guide, record the deciding fact in `evidence` for every PR so each label can be audited from the released sheet, and report every rate with and without `unclear`. Threats to Validity names the single coder and states that no agreement statistic is reported.
- **Main estimate:** the share of PRs labelled agent, with a 95% Wilson score interval, the interval recommended for a binomial proportion by Brown, Cai and DasGupta (2001). Also report agent + mixed, which counts any agent-written code.
- **By group:** supervised assistants (Copilot, Codex, Cursor, Claude Code) and Devin separately, with Wilson intervals; per-tool rates are not powered (Sample).
- **Use in the paper:** state the rate in Methodology and in Threats to Validity. If PRs labelled human are present, rerun the RQ1 primary model without them as a sensitivity analysis.
- **Human baseline:** the audit does not check human PRs for unmarked AI assistance (Reviewer 3). The paper names that as a threat and calls the groups "agent-attributed" and "not attributed".

## Sources

Checked online on 2026-11-02 unless marked; each entry still needs your check before it goes into the bib.

- C. Bird, N. Nagappan, B. Murphy, H. Gall and P. Devanbu, "Don't touch my code! Examining the effects of ownership on software quality," ESEC/FSE 2011. (from memory, not checked online)
- L. D. Brown, T. T. Cai and A. DasGupta, "Interval estimation for a binomial proportion," Statistical Science 16(2), 2001. [doi:10.1214/ss/1009213286](https://www.doi.org/10.1214/SS/1009213286)
- W. G. Cochran, *Sampling Techniques*, 3rd ed., Wiley, 1977. (standard text, from memory)
- D. Kawrykow and M. P. Robillard, "Non-essential changes in version histories," ICSE 2011. [paper](https://www.cs.mcgill.ca/~martin/papers/icse2011.pdf)
- H. Li, H. Zhang and A. E. Hassan, "The rise of AI teammates in software engineering (SE) 3.0: How autonomous coding agents are reshaping software engineering," arXiv:2507.15003, 2025. [paper](https://arxiv.org/html/2507.15003v1)
- N. McDonald, S. Schoenebeck and A. Forte, "Reliability and inter-rater reliability in qualitative research: Norms and guidelines for CSCW and HCI practice," Proc. ACM Hum.-Comput. Interact. 3(CSCW), 2019. [doi:10.1145/3359174](https://doi.org/10.1145/3359174)
- M. F. Rabbi, A. K. Turzo, A. I. Champa and M. F. Zibran, "Insights into security-related AI-generated pull requests," arXiv:2604.19965, 2026 (venue reportedly EASE 2026, not confirmed). [paper](https://arxiv.org/html/2604.19965v1)
- P. Ralph et al., "ACM SIGSOFT Empirical Standards," arXiv:2010.03525, 2021 (Inter-Rater Reliability supplement; checked via full text on 2026-11-02).
- R. Purushothaman and D. E. Perry, "Toward understanding the rhetoric of small source code changes," IEEE TSE 31(6), 2005, pp. 511–526. [record](https://ftp.math.utah.edu/pub/tex/bib/idx/ieeetranssoftweng2000/31/6/511_526.html)
