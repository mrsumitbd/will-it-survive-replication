# Paper numbers: reproduced vs. reported

Results directory: `expected_results/` (mode: expected (shipped outputs of the paper run)).

**648 of 651 reproducible numbers match**, grouped in 207 statements: **206 of 207 statements match** (mismatch: 0, missing: 0, partial: 1; 0 of the matches need double rounding, see below). Not counted: 17 external, 0 pending.

An entry groups the numbers of one statement; it matches when every number matches after rounding to the precision printed in the paper. Status: match, mismatch, missing (output absent), partial (agrees, but not independently reproducible from the shipped data), external (AIDev release, literature or a design choice), pending (second rater).


## Abstract

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Abstract | lines, PRs, repositories of the primary design | 648,959 lines, 2,794 PRs, 215 repositories | 648,959; 2,794; 215 | match |  |
| Abstract | agent HR, primary, with 95% CI | HR = 0.83 [0.75, 0.93] | 0.8347; 0.7490; 0.9301 | match |  |
| Abstract | supervised-assistant HR | HR = 0.76 | 0.7566 | match |  |
| Abstract | Devin HR | HR = 0.95 | 0.9533 | match |  |
| Abstract | after mid-May 2025, supervised and Devin code modified later | both upper CI < 1 | sup 0.731 [0.582, 0.918], Devin 0.608 [0.441, 0.838] | match |  |
| Abstract | without PR weighting the difference is not significant | agent p >= 0.05 (unweighted) | p = 0.495 | match |  |
| Abstract | Cramer's V (RQ2) | V = 0.080 | 0.07999 | match |  |
| Abstract | AUC-ROC size/history baseline vs. token model (agent files) | AUC-ROC 0.78 vs. 0.70 | 0.7848; 0.6984 | match | token model = random forest, the best model by AUC-PR |
| Abstract | Macro F1 of RQ3b vs. random guess | Macro F1 of 0.31 against 0.25 | 0.3124; 0.2455 | match | first-rank models: XGBoost 0.316, CatBoost 0.313, random forest 0.312; random guess = expected Macro F1 of a uniform guess over the four bins |

## 1 Introduction

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 1 | agent PRs in AIDev | 932,791 PRs | 932,791 | external | AIDev release (Li et al.) (consistent) |
| Sec. 1 | Stack Overflow survey 2024 | 76% of respondents |  | external | literature |
| Sec. 1 | GitClear changed lines | about 153 million changed lines from 2020 to 2023; less than two weeks |  | external | literature |
| Sec. 1 | mined cohort | 2,802 merged PRs, 1,470 agent-attributed and 1,332 human, in 215 repositories | 2,802; 1,470; 1,332; 215 | match |  |
| Sec. 1 (RQ1) | supervised-assistant HR | HR = 0.76 [0.65, 0.87] | 0.7566; 0.6546; 0.8746 | match |  |
| Sec. 1 (RQ1) | Devin HR | HR = 0.95 | 0.9533 | match |  |
| Sec. 1 (RQ2) | Cramer's V | V = 0.080 | 0.07999 | match |  |
| Sec. 1 (RQ2) | perfective, corrective, adaptive, preventive, other shares (agent vs. human) | 38.3% vs. 35.4%; 35.6% vs. 32.5%; 11.7% vs. 12.7%; 3.9% vs. 3.5%; 10.6% vs. 15.9% | 38.304; 35.352; 35.577; 32.483; 11.674; 12.722; 3.877; 3.533; 10.567; 15.910 | match |  |
| Sec. 1 (RQ3) | token model AUC-ROC | AUC-ROC 0.70 | 0.6984 | match |  |
| Sec. 1 (RQ3) | baseline AUC-ROC | AUC-ROC 0.78 | 0.7848 | match |  |
| Sec. 1 (RQ3) | Macro F1 vs. chance | Macro F1 0.31 against 0.25 | 0.3124; 0.2455 | match |  |

## 2.1 Dataset

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 2.1.1 | AIDev agent PRs per agent | 932,791: 814,522 Codex, 50,447 Copilot, 32,941 Cursor, 29,744 Devin, 5,137 Claude Code | 932,791; 814,522; 50,447; 32,941; 29,744; 5,137 | external | AIDev release; checked against data/aidev/aidev_summary.json (consistent) |
| Sec. 2.1.1 | AIDev repositories | 116,211 repositories | 116,211 | external | AIDev release (distinct repo_id); data/aidev/aidev_summary.json (consistent) |
| Sec. 2.1.1 | AIDev collection end | up to August 1, 2025 |  | external | AIDev release |
| Sec. 2.1.1 | AIDev human PRs (created between January and June 2025) | 6,618 human PRs in 818 repositories | 6,618; 818 | external | AIDev release; data/aidev/aidev_summary.json (consistent) |

## Table 1 (funnel)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Table 1 | AIDev, all repositories | 116,211 \| 932,791 \| 6,618 | 116,211; 932,791; 6,618 | external | AIDev release; data/aidev/aidev_summary.json (consistent) |
| Table 1 | (1) repositories with agent and human PRs | 816 \| 9,797 \| 6,598 | 816; 9,797; 6,598 | match |  |
| Table 1 | (2) software license | 666 \| 7,625 \| 5,054 | 666; 7,625; 5,054 | match |  |
| Table 1 | (3) not archived, release or tag, >=2 contributors | 566 \| 6,891 \| 4,559 | 566; 6,891; 4,559 | match |  |
| Table 1 | (4) bottom-quartile removal per language | 266 \| 4,649 \| 3,196 | 266; 4,649; 3,196 | match |  |
| Table 1 | (5) code ratio within the 97% interval | 256 \| 4,563 \| 3,132 | 256; 4,563; 3,132 | partial | the per-repository code ratios of the final run are not available, so step (5) is reproduced from the cohort list (246 repositories with merged PRs + 10 Q1 repositories without any merged PR); the 10 removed repositories are inferred |
| Table 1 | repositories with at least one merged PR | 246 \| 2,302 \| 2,386 | 246; 2,302; 2,386 | match |  |
| Table 1 | repository history available | 244 \| 2,266 \| 2,351 | 244; 2,266; 2,351 | match |  |
| Table 1 | not listed in both AIDev tables | 244 \| 2,261 \| 2,346 | 244; 2,261; 2,346 | match |  |
| Table 1 | merge commit located | 241 \| 2,092 \| 2,156 | 241; 2,092; 2,156 | match |  |
| Table 1 | adds at least one tracked source line (mined cohort) | 215 \| 1,470 \| 1,332 | 215; 1,470; 1,332 | match |  |
| Table 1 note | human PRs in repositories without agent PRs | 20 in two (2) repositories | 20; 2 | external | AIDev release; data/aidev/aidev_summary.json (consistent) |

## 2.1.3 Final cohort

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 2.1.3 | filtered repositories, with a merged PR | 256 filtered repositories, 246 have at least one merged PR | 256; 246 | match |  |
| Sec. 2.1.3 | two repositories cannot be mined | dust-tt/dust and elizaos/eliza | dust-tt/dust, elizaos/eliza | match |  |
| Sec. 2.1.3 | PRs listed as both agent and human PRs | five (5) PRs | 5 | match |  |
| Sec. 2.1.3 | merge commit not located | 359 PRs | 359 | match |  |
| Sec. 2.1.3 | PRs that change no tracked source file | 1,446 PRs | 1,446 | match |  |
| Sec. 2.1.3 | mined cohort | 2,802 merged PRs in 215 repositories | 2,802; 215 | match |  |
| Sec. 2.1.3 | agent PRs by agent | 1,470 (492 Copilot, 491 Codex, 412 Devin, 44 Cursor, 31 Claude Code) and 1,332 human | 1,470; 492; 491; 412; 44; 31; 1,332 | match |  |
| Sec. 2.1.3 | PRs with only vendored or generated lines; primary design PRs | Eight (8) PRs; 2,794 PRs | 8; 2,794 | match |  |

## 2.1.5 Attribution validation

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 2.1.5 | merged agent PRs with a located merge commit | 2,092 | 2,092 | match |  |
| Sec. 2.1.5 | audited PRs (data/audit/audit_rater1.csv) | 325 PRs | 325 | match |  |
| Sec. 2.1.5 | sample size for a proportion, 95% confidence, +-5% margin, finite-population correction (N = 2,092) | 325 | 324.68, rounded up 325 | match |  |
| Sec. 2.1.5 | PRs in the second-rater sheet | 100 of the 325 PRs (31%) | 100; 325; 30.769 | match |  |
| Sec. 2.1.5 | typical share for agreement; PRs needed | 10 to 25%; 62 PRs; kappa = 0.8 vs. 0.5 |  | external | literature |
| Sec. 2.1.5 | Cohen's kappa between the two raters | kappa = 0.85 | 0.8504 | match |  |
| Sec. 2.1.5 | agent label | 218 (67.1% [61.8, 72.0]) | 218; 67.077; 61.794; 71.961 | match |  |
| Sec. 2.1.5 | mixed, human, unclear labels | 71 (21.8%), 20 (6.2%), 16 (4.9%) | 71; 21.846; 20; 6.154; 16; 4.923 | match |  |
| Sec. 2.1.5 | agent made substantive changes (agent or mixed) | 88.9% | 88.923 | match |  |
| Sec. 2.1.5 | agent made substantive changes, mined cohort | 87.3% [82.3, 91.1] | 87.330; 82.298; 91.087 | match |  |
| Sec. 2.1.5 | Devin | 98.4% of the 64 Devin PRs | 98.438; 64 | match |  |
| Sec. 2.1.5 | supervised assistants | 82.8% [76.1, 87.9] of the 157 | 82.803; 76.135; 87.903; 157 | match |  |
| Sec. 2.1.5 | labeled human, mined cohort | 16 PRs | 16 | match |  |

## 2.2 Measuring survival

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 2.2.1 | end of the observation window | December 31, 2025 |  | external | design choice |
| Sec. 2.2.1 | every unit is observed for at least five months | minimum follow-up >= 150 days | last merge 2025-07-30, minimum follow-up 154.2 days | match |  |
| Sec. 2.2.3 | substantial modification thresholds | theta in {25%, 50%, 75%, 100%} |  | external | design choice |
| Sec. 2.2.5 | largest human PR | adds more than 34,000 lines outside vendored and generated paths | 34,380 lines outside vendored paths | match |  |
| Sec. 2.2.5 | illustrative example (not data) | 10,000 of the 10,990 human lines (91%); one of the 100 PRs (1%) | 10,000; 10,990; 90.992; 100; 1 | match | arithmetic of a hypothetical example |

## Table 2 (units and events)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Table 2 | units | 648,959 \| 60,095 \| 18,120 | 648,959; 60,095; 18,120 | match |  |
| Table 2 | modified, human | 34.5% (30.1%) \| 44.8% (39.3%) \| 88.2% (85.2%) | 34.512; 30.100; 44.771; 39.303; 88.183; 85.196 | match |  |
| Table 2 | modified, supervised assistants | 24.1% (23.3%) \| 34.5% (30.9%) \| 81.3% (77.1%) | 24.126; 23.337; 34.451; 30.914; 81.312; 77.082 | match |  |
| Table 2 | modified, Devin | 38.0% (23.6%) \| 52.5% (51.1%) \| 91.5% (85.6%) | 37.953; 23.637; 52.512; 51.100; 91.512; 85.624 | match |  |
| Table 2 | median days to modification (modified units), PR-weighted (unweighted) | 40.1 (39.0) \| 35.9 (40.1) \| 10.0 (11.6) | 40.127; 38.993; 35.895; 40.073; 9.993; 11.568 | match |  |
| Table 2 | median follow-up of censored units (days) | 218.3 \| 236.5 \| 216.5 | 218.266; 236.463; 216.534 | match |  |
| Table 2 note | PRs and lines of the primary design | 2,794 PRs; 648,959 lines | 2,794; 648,959 | match |  |
| Table 2 note | Devin's largest PR | adds 41.6% of its lines | 41.618 | match |  |

## 3 RQ1

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.2 | RMST horizons and bootstrap | tau = 90 and 150 days; 1,000 repository bootstrap replicates | 90; 150; 1,000 | external | design choice; replicates read from results/rq1/rq1_line.json (consistent) |
| Sec. 3.2 | sampled design | at most 20 source files per PR and 100 random lines per file |  | external | design choice |
| Sec. 3.2 | PRs with an AIDev task type | 2,781 of our 2,794 PRs | 2,781; 2,794 | match |  |
| Sec. 3.2 | modifying commits that belong to an AIDev PR | only 8.5% of the 5,600 commits | 8.464; 5,600 | match |  |
| Sec. 3.3 | agent HR, primary | HR = 0.83 [0.75, 0.93], p = 0.001 | 0.8347; 0.7490; 0.9301; 0.00107 | match |  |
| Sec. 3.3 | supervised-assistant HR | HR = 0.76 [0.65, 0.87] | 0.7566; 0.6546; 0.8746 | match |  |
| Sec. 3.3 | supervised-assistant p | p < 0.001 | 0.00016 | match |  |
| Sec. 3.3 | Devin HR | HR = 0.95 [0.82, 1.11], p = 0.53 | 0.9533; 0.8205; 1.1075; 0.5318 | match |  |
| Sec. 3.3 | hazard of supervised-assistant lines lower by | 24% lower | 24.335 | match |  |
| Sec. 3.3 | modified by the end of the window, agent vs. human (PR-weighted) | 28.0% and 34.5% | 28.009; 34.512 | match |  |
| Sec. 3.3 | difference in percentage points [bootstrap CI] | -6.5 pp [-10.7, -1.7] | -6.503; -10.679; -1.655 | match |  |

## Table 3 (RQ1 designs)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Table 3 | Line, primary | 648,959 \| 28.0 / 34.5 \| 0.83 [0.75, 0.93] \| 0.76 [0.65, 0.87] \| 0.95 [0.82, 1.11] | 648,959; 28.009; 34.512; 0.8347; 0.7490; 0.9301; 0.7566; 0.6546; 0.8746; 0.9533; 0.8205; 1.1075 | match |  |
| Table 3 | Line, with vendored paths | 854,139 \| 28.1 / 34.8 \| 0.83 [0.75, 0.93] \| 0.76 [0.66, 0.87] \| 0.95 [0.82, 1.10] | 854,139; 28.127; 34.756; 0.8332; 0.7484; 0.9276; 0.7580; 0.6569; 0.8747; 0.9482; 0.8166; 1.1010 | match |  |
| Table 3 | Line, unweighted | 648,959 \| 23.4 / 30.1 \| 0.91 [0.69, 1.20] \| 0.80 [0.60, 1.07] \| 1.06 [0.65, 1.71] | 648,959; 23.436; 30.100; 0.9076; 0.6870; 1.1990; 0.7991; 0.5955; 1.0722; 1.0590; 0.6546; 1.7133 | match |  |
| Table 3 | Line, unweighted, with vendored | 854,139 \| 23.3 / 50.3 \| 0.92 [0.68, 1.25] \| 0.86 [0.58, 1.29] \| 1.00 [0.68, 1.48] | 854,139; 23.281; 50.333; 0.9230; 0.6836; 1.2461; 0.8633; 0.5780; 1.2894; 0.9995; 0.6770; 1.4755 | match |  |
| Table 3 | Line, sampled | 277,283 \| 28.1 / 34.7 \| 0.83 [0.75, 0.93] \| 0.76 [0.66, 0.88] \| 0.95 [0.81, 1.10] | 277,283; 28.140; 34.718; 0.8338; 0.7489; 0.9285; 0.7600; 0.6580; 0.8778; 0.9452; 0.8146; 1.0968 | match |  |
| Table 3 | Hunk, primary | 60,095 \| 39.5 / 44.8 \| 0.93 [0.85, 1.02] \| 0.82 [0.72, 0.92] \| 1.11 [0.98, 1.26] | 60,095; 39.523; 44.771; 0.9304; 0.8493; 1.0193; 0.8176; 0.7245; 0.9227; 1.1134; 0.9824; 1.2617 | match |  |
| Table 3 | File, primary | 18,120 \| 84.2 / 88.2 \| 0.97 [0.90, 1.04] \| 0.95 [0.86, 1.04] \| 1.01 [0.90, 1.13] | 18,120; 84.177; 88.183; 0.9701; 0.9024; 1.0430; 0.9473; 0.8637; 1.0389; 1.0083; 0.9034; 1.1253 | match |  |
| Table 3 | bold cells | bold exactly where p < 0.05 | same cells | match |  |
| Table 3 | Line, repository x task type | 648,959 \| 28.0 / 34.5 \| 0.87 [0.78, 0.99] \| 0.77 [0.65, 0.91] \| 1.03 [0.87, 1.21] | 648,959; 28.009; 34.512; 0.8746; 0.7750; 0.9869; 0.7676; 0.6504; 0.9058; 1.0266; 0.8717; 1.2090 | match |  |
| Table 3 | bold cells, repository x task type row | agent and supervised bold, Devin not | 0.030, 0.002, 0.753 | match |  |

## Figure 1 (KM curves)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Figure 1 | figure file | results/figures/rq1_km_line_primary.pdf | present | match |  |
| Figure 1 caption | no supervised-assistant line at risk after | day 281 | 280.711 | match |  |
| Figure 1 caption | largest steps (supervised assistants, Devin) | day 223 and day 313 | 223.376; 313.070 | match |  |
| Figure 1 caption | each step comes from a single PR | share of the step from its top PR = 1 | 1.000, 1.000 | match |  |

## Figure 2 (HR across design choices)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Figure 2 | figure file | results/figures/rq1_hr_forest.pdf | present | match |  |

## 3.3 Weighting, vendored code and pooling

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | share of lines in vendored paths: human, supervised, Devin | 34.1%, 3.8%, 0.3% | 34.115; 3.815; 0.292 | match |  |
| Sec. 3.3 | without weighting, no difference is significant | all unweighted line HRs p >= 0.05 | 0.49, 0.13, 0.82, 0.60, 0.47, 1.00 | match |  |
| Sec. 3.3 | pooled Cox, primary design | HR = 0.89 [0.80, 1.00] | 0.8942; 0.8027; 0.9962 | match |  |
| Sec. 3.3 | pooled Cox, unweighted lines with vendored paths | 0.76 [0.52, 1.10] | 0.7551; 0.5205; 1.0955 | match |  |

## 3.3 Proportional hazards and RMST

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | Schoenfeld test (unweighted model) | p < 0.001 | 5.4e-174 | match | the weighted test is not usable (fractional weights are treated as frequencies); the unweighted model test is reported |
| Sec. 3.3 | agent HR in days 0-14, 14-90, >90 | 1.01 [0.83, 1.23], 0.85 [0.72, 1.01], 0.65 [0.53, 0.80] | 1.0088; 0.8300; 1.2263; 0.8526; 0.7202; 1.0093; 0.6518; 0.5318; 0.7989 | match |  |
| Sec. 3.3 | supervised HR in days 0-14, 14-90, >90 | 1.10 [0.84, 1.44], 0.79 [0.64, 0.99], 0.47 [0.36, 0.61] | 1.1005; 0.8404; 1.4411; 0.7938; 0.6383; 0.9872; 0.4713; 0.3634; 0.6113 | match |  |
| Sec. 3.3 | Devin does not differ in any period | Devin p >= 0.05 in every period | 0.47, 0.63, 0.85 | match |  |
| Sec. 3.3 | RMST difference at 90 days | +0.8 days [-2.3, 3.8] | 0.777; -2.311; 3.826 | match |  |
| Sec. 3.3 | RMST difference at 150 days | +2.2 days [-3.4, 7.9] | 2.194; -3.443; 7.872 | match |  |
| Sec. 3.3 | modified within 14 days (PR-weighted): supervised, Devin, human | 8.3%, 12.0%, 9.6% | 8.298; 11.952; 9.556 | match |  |

## 3.3 Calendar time

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | median merge dates | June 18, 2025 (supervised) vs. April 9 (human and Devin) | 2025-06-18, 2025-04-09, 2025-04-09 | match |  |
| Sec. 3.3 | calendar cut | May 16, 2025 (release of Codex) |  | external | design choice (product documentation) |
| Sec. 3.3 | PRs merged on or after May 16: agent, supervised, Devin HR | 0.70 [0.57, 0.84], 0.73 [0.58, 0.92], 0.61 [0.44, 0.84] | 0.6960; 0.5749; 0.8426; 0.7310; 0.5822; 0.9178; 0.6079; 0.4407; 0.8385 | match |  |
| Sec. 3.3 | Devin PRs after the cut | 133 Devin PRs | 133 | match |  |
| Sec. 3.3 | strata repository x merge month: supervised HR | 0.96 [0.74, 1.24] | 0.9610; 0.7420; 1.2446 | match |  |
| Sec. 3.3 | strata with agent and human PRs | 128 of the 657 strata | 128; 657 | match |  |

## 3.3 Hunks, files and substantial modification

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | supervised-assistant hunks | HR = 0.82 | 0.8176 | match |  |
| Sec. 3.3 | agent group and Devin hunks not significant | p >= 0.05 | 0.121, 0.092 | match |  |
| Sec. 3.3 | Devin hunks, unweighted | HR = 1.24 [1.00, 1.53], p = 0.049 | 1.2361; 1.0006; 1.5271; 0.04940 | match |  |
| Sec. 3.3 | at the file level, no group differs | all file HRs p >= 0.05 | 0.41, 0.25, 0.88 | match |  |
| Sec. 3.3 | substantial modification: agent HR range | HR 0.80 to 0.86 | 0.8037; 0.8598 | match |  |
| Sec. 3.3 | substantial modification: agent p | all p < 0.02 | max p 0.0177 | match |  |
| Sec. 3.3 | substantial modification: supervised HR range | 0.72 to 0.77 | 0.7194; 0.7651 | match |  |
| Sec. 3.3 | substantial modification: Devin HR range | 0.93 to 1.01 | 0.9276; 1.0094 | match |  |
| Sec. 3.3 | Devin does not differ at any theta | Devin p >= 0.05 | min p 0.351 | match |  |

## 3.3 Attribution, file history and repository activity

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | cohort PRs labeled human; human or unclear | 16; 28 | 16; 28 | match |  |
| Sec. 3.3 | agent HR without them | 0.84 and 0.83 | 0.8352; 0.8309 | match |  |
| Sec. 3.3 | PRs labeled agent; HR | 136 audited PRs; 0.89 [0.64, 1.23] | 136; 0.8886; 0.6402; 1.2335 | match |  |
| Sec. 3.3 | with file-history covariates: agent, supervised HR | 0.88 [0.79, 0.98]; 0.81 [0.70, 0.93] | 0.8787; 0.7889; 0.9788; 0.8066; 0.6987; 0.9312 | match |  |
| Sec. 3.3 and 5.1 | share of the difference explained by the files | about a quarter | 27%, 29% (1 - HR scale, log scale) | match |  |
| Sec. 3.3 | covariate HRs: recent changes, test file | HR = 1.26 per log unit; HR = 0.73 | 1.2610; 0.7286 | match |  |
| Sec. 3.3 | activity terciles (AIDev PR count): agent HR | 0.83 [0.44, 1.57], 0.63 [0.43, 0.94], 0.86 [0.77, 0.97] | 0.8318; 0.4402; 1.5716; 0.6338; 0.4276; 0.9394; 0.8632; 0.7703; 0.9673 | match |  |
| Sec. 3.3 | same direction in every tercile | all HR < 1 | yes | match |  |

## 3.3 Task type

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | task-type mix, Cramer's V | Cramer's V = 0.12 | 0.1171 | match |  |
| Sec. 3.3 | fixes: supervised vs. human PRs; features: Devin vs. human PRs | 41.3%, 35.0%; 47.3%, 39.5% | 41.327; 34.966; 47.330; 39.488 | match |  |
| Sec. 3.3 | repository x task type: supervised, agent, Devin HR | 0.77 [0.65, 0.91]; 0.87 [0.78, 0.99]; 1.03 [0.87, 1.21] | 0.7676; 0.6504; 0.9058; 0.8746; 0.7750; 0.9869; 1.0266; 0.8717; 1.2090 | match |  |
| Sec. 3.3 | repository x task type, unweighted: supervised HR | 0.77 [0.61, 0.96] | 0.7670; 0.6107; 0.9633 | match |  |
| Sec. 3.3 | unweighted supervised difference significant within repository x task type | p < 0.05 | 0.023 | match |  |
| Sec. 3.3 | supervised HR within fixes; within features | 0.75 [0.57, 0.98]; 0.82 [0.63, 1.06] | 0.7476; 0.5697; 0.9810; 0.8184; 0.6329; 1.0583 | match |  |

## 3.3 Individual agents

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 3.3 | Copilot and Codex HR | HR = 0.75; HR = 0.69 | 0.7462; 0.6933 | match |  |
| Sec. 3.3 | PRs of Copilot and Codex | 980 of their 1,055 PRs | 980; 1,055 | match |  |
| Sec. 3.3 | PRs of Cursor and Claude Code | 44 and 31 PRs | 44; 31 | match |  |
| Sec. 3.3 | share of lines added by the largest PR, Cursor and Claude Code | 26% and 33% | 26.403; 32.870 | match |  |

## Table 4 (per agent)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Table 4 | GitHub Copilot | 489 \| 0.75 [0.59, 0.94] \| 22.5 [19.6, 25.3] \| 9.7% | 489; 0.7462; 0.5948; 0.9361; 22.453; 19.616; 25.281; 9.686 | match |  |
| Table 4 | OpenAI Codex | 491 \| 0.69 [0.57, 0.84] \| 24.4 [21.4, 27.8] \| 5.7% | 491; 0.6933; 0.5701; 0.8432; 24.412; 21.419; 27.823; 5.659 | match |  |
| Table 4 | Cursor | 44 \| 1.81 [0.91, 3.60] \| 35.0 [24.6, 46.7] \| 26.4% | 44; 1.8073; 0.9069; 3.6018; 35.044; 24.598; 46.653; 26.403 | match |  |
| Table 4 | Claude Code | 31 \| 1.66 [0.81, 3.41] \| 30.5 [18.0, 43.5] \| 32.9% | 31; 1.6641; 0.8128; 3.4071; 30.478; 18.004; 43.533; 32.870 | match |  |
| Table 4 | Devin | 412 \| 0.96 [0.82, 1.11] \| 38.0 [34.4, 41.5] \| 41.6% | 412; 0.9567; 0.8234; 1.1117; 37.953; 34.382; 41.462; 41.618 | match |  |
| Table 4 | Human | 1,327 \| 34.5 [32.6, 36.6] \| 9.0% | 1,327; 34.512; 32.612; 36.576; 8.999 | match |  |
| Table 4 | bold cells | bold exactly where p < 0.05 (Copilot, Codex) | Copilot p=0.0114, OpenAI_Codex p=0.000244, Cursor p=0.0925, Claude_Code p=0.164, Devin p=0.564 | match |  |

## 4 RQ2

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 4.2 | merge commits among the modifying commits; their observations in the adaptive class | 416 merge commits; 280 of their 456 observations | 416; 280; 456 | match |  |
| Sec. 4.3 | observations | 6,112 (2,347 agent, 3,765 human) | 6,112; 2,347; 3,765 | match |  |
| Sec. 4.3 | chi-square test | chi2 = 39.1, df = 4 | 39.104; 4 | match |  |
| Sec. 4.3 | p-value | p < 0.001 | 6.6e-08 | match |  |
| Sec. 4.3 | Cramer's V | V = 0.080 | 0.07999 | match |  |
| Sec. 4.3 | only the Other class deviates beyond \|z\| > 2 | Other only | {'Adaptive': -0.89, 'Corrective': 1.59, 'Other': -4.28, 'Perfective': 1.46, 'Preventive': 0.54} | match |  |
| Sec. 4.3 | Other share; z | 10.6% vs. 15.9%, z = -4.28 | 10.567; 15.910; -4.2833 | match |  |
| Sec. 4.3 | Corrective share; z | 35.6% vs. 32.5%; z = +1.59 | 35.577; 32.483; 1.5912 | match |  |
| Sec. 4.3 | Perfective share | 38.3% vs. 35.4% | 38.304; 35.352 | match |  |

## Table 5 (RQ2 intent)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Table 5 | Corrective | 35.6 \| 32.5 \| +3.1 \| +1.59 | 35.577; 32.483; 3.100; 1.5912 | match | Delta = difference of the rounded shares, as in the paper |
| Table 5 | Perfective | 38.3 \| 35.4 \| +2.9 \| +1.46 | 38.304; 35.352; 2.900; 1.4586 | match | Delta = difference of the rounded shares, as in the paper |
| Table 5 | Adaptive | 11.7 \| 12.7 \| -1.0 \| -0.89 | 11.674; 12.722; -1.000; -0.8910 | match | Delta = difference of the rounded shares, as in the paper |
| Table 5 | Preventive | 3.9 \| 3.5 \| +0.4 \| +0.54 | 3.877; 3.533; 0.400; 0.5374 | match | Delta = difference of the rounded shares, as in the paper |
| Table 5 | Other | 10.6 \| 15.9 \| -5.3 \| -4.28 | 10.567; 15.910; -5.300; -4.2833 | match | Delta = difference of the rounded shares, as in the paper |
| Table 5 note | test statistics | chi2 = 39.1, df = 4, V = 0.080 | 39.104; 4; 0.07999 | match |  |

## 4.3 Sensitivity analyses

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 4.3 | without merge commits | V = 0.081 | 0.08100 | match |  |
| Sec. 4.3 | without merge commits: adaptive share (agent, human) | 8.3% (agent) and 8.4% (human) | 8.296; 8.406 | match |  |
| Sec. 4.3 | single-group commits | V = 0.084, Corrective 36.2% vs. 32.4% | 0.08392; 36.240; 32.370 | match |  |
| Sec. 4.3 | file-commit unit | chi2 = 7.0, p = 0.13, V = 0.035 | 7.017; 0.1350; 0.03469 | match |  |
| Sec. 4.3 | substring matching: effect size | V = 0.155 | 0.15466 | match |  |
| Sec. 4.3 | substring matching: preventive share agent, human | 29.7% and 18.8% | 29.740; 18.805 | match |  |
| Sec. 4.3 | whole-word matching: preventive share agent, human | 3.9% and 3.5% | 3.877; 3.533 | match |  |

## 5 RQ3 approach

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 5.2 | 30 repetitions of 10-fold cross-validation | 300 scores per model | RQ3a [300], RQ3b [300] | match |  |
| Sec. 5.2 | explanation folds; vocabulary | five repository folds; at most 1,000 features; unigrams to trigrams |  | external | design choice |
| Sec. 5.2 (RQ3b) | smallest bin | the smallest bin holds 15% of the files | 15.116 | match |  |
| Sec. 5.2 (RQ3b) | files dropped for missing author history | 22 of the 6,300 modified agent files, 0.3% | 22; 6,300; 0.350 | match |  |
| Sec. 5.2 (RQ3b) | chance level of Macro F1 with four bins | 0.25 | 0.2455 | match |  |

## Table 6 (RQ3 models)

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Table 6 | Random forest | 0.70 (1) \| 0.89 (1) \| 0.31 (1) \| 0.60 (1) | 0.6984; 1; 0.8895; 1; 0.3124; 1; 0.6010; 1 | match |  |
| Table 6 | CatBoost | 0.70 (1) \| 0.89 (1) \| 0.31 (1) \| 0.61 (1) | 0.6955; 1; 0.8883; 1; 0.3133; 1; 0.6063; 1 | match |  |
| Table 6 | XGBoost | 0.69 (1) \| 0.89 (1) \| 0.32 (1) \| 0.60 (1) | 0.6936; 1; 0.8860; 1; 0.3159; 1; 0.6049; 1 | match |  |
| Table 6 | MLP | 0.66 (2) \| 0.87 (2) \| 0.28 (2) \| 0.58 (2) | 0.6624; 2; 0.8748; 2; 0.2795; 2; 0.5811; 2 | match |  |
| Table 6 | SVM | 0.64 (3) \| 0.87 (2) \| 0.28 (2) \| 0.58 (2) | 0.6405; 3; 0.8741; 2; 0.2790; 2; 0.5771; 2 | match |  |
| Table 6 | Logistic regression | 0.63 (3) \| 0.87 (2) \| 0.28 (2) \| 0.58 (2) | 0.6333; 3; 0.8699; 2; 0.2807; 2; 0.5813; 2 | match |  |
| Table 6 | KNN | 0.64 (3) \| 0.85 (3) \| 0.27 (2) \| 0.53 (4) | 0.6361; 3; 0.8497; 3; 0.2707; 2; 0.5349; 4 | match |  |
| Table 6 | Naive Bayes | 0.51 (4) \| 0.82 (4) \| 0.28 (2) \| 0.56 (3) | 0.5144; 4; 0.8206; 4; 0.2761; 2; 0.5618; 3 | match |  |
| Table 6 | size and history baseline (LR) | 0.78 \| 0.93 | 0.7848; 0.9292 | match |  |
| Table 6 | random guess | 0.80 (AUC-PR, share of modified files 79.7%) \| 0.25 (Macro F1) | 0.7975; 79.748; 0.2455 | match |  |

## 5.3 RQ3a results

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 5.3 | agent dataset | 6,661 files from 155 repositories, 79.7% modified | 6,661; 155; 79.748 | match |  |
| Sec. 5.3 | first Scott-Knott rank (AUC-PR) | random forest, CatBoost and XGBoost | CatBoost, RandomForest, XGBoost | match |  |
| Sec. 5.3 | first-rank AUC-ROC | about 0.70 | 0.6984, 0.6955, 0.6936 | match |  |
| Sec. 5.3 | first-rank AUC-PR | about 0.89 | 0.8895, 0.8883, 0.886 | match |  |
| Sec. 5.3 | AUC-PR of a random classifier | about 80% of the files are modified | 79.748 | match |  |
| Sec. 5.3 | logistic regression baseline | AUC-ROC 0.78, AUC-PR 0.93 | 0.7848; 0.9292 | match |  |
| Sec. 5.3 | all files | 15,946 files (83.8% modified) | 15,946; 83.802 | match |  |
| Sec. 5.3 | all files: random forest, XGBoost, CatBoost AUC-ROC | about 0.69 | 0.6973, 0.6938, 0.693 | match |  |
| Sec. 5.3 | all files: baseline AUC-ROC | 0.75 | 0.7491 | match |  |
| Sec. 5.3 | tokens with the largest SHAP values | key, model, config, error, state | key, model, config, error, state | match |  |
| Sec. 5.3 | LIME top tokens among the 50 highest SHAP | 46.9% | 46.900 | match |  |
| Sec. 5.3 | LIME runtime per file | about 0.2 seconds | 0.210 | match | machine dependent |

## 5.3 RQ3a line localization

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 5.3 | localization data | 1,941 files from 109 repositories; 163,409 lines, 22.8% modified | 1,941; 109; 163,409; 22.783 | match |  |
| Sec. 5.3 | SHAP ranking AUC-ROC (pooled, per-file mean); line length | AUC-ROC 0.546, per-file mean 0.576; length 0.552 | 0.54649; 0.57583; 0.55232 | match |  |
| Sec. 5.3 | precision in the top 10% of lines: SHAP, random, longest lines | 36.7%, 30.1%, 39.7% | 36.657; 30.062; 39.729 | match |  |

## 5.3 RQ3b results

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| Sec. 5.3 | agent dataset | 6,278 modified files from 160 repositories | 6,278; 160 | match |  |
| Sec. 5.3 | bins | 15.1% within one day, 25.0% within one week, 26.8% within one month, 33.1% later | 15.116; 24.992; 26.808; 33.084 | match |  |
| Sec. 5.3 | AutoSpearman keeps | all eight (8) features | 8 | match |  |
| Sec. 5.3 | first Scott-Knott rank (Macro F1) | XGBoost, CatBoost and random forest | CatBoost, RandomForest, XGBoost | match |  |
| Sec. 5.3 | first-rank Macro F1 | about 0.31 | 0.3159, 0.3133, 0.3124 | match |  |
| Sec. 5.3 | first-rank macro AUC-ROC | about 0.60 | 0.6049, 0.6063, 0.601 | match |  |
| Sec. 5.3 | out of sample, XGBoost | 0.32 and 0.61 | 0.3204; 0.6050 | match |  |
| Sec. 5.3 | all files: first Scott-Knott rank (Macro F1) | CatBoost alone | CatBoost | match |  |
| Sec. 5.3 | all files: CatBoost | Macro F1 0.32, AUC-ROC 0.62 | 0.3226; 0.6153 | match |  |
| Sec. 5.3 | SHAP order | file changes (3 months) first, open PRs second, file age third | file_modification_frequency_3m, project_open_prs_at_birth, file_age_days | match |  |
| Sec. 5.3 | LIME agrees on the leading feature | file changes (3 months) | file_modification_frequency_3m | match |  |
| Sec. 5.3 | PR-author history ranks | fifth and sixth of the eight features | 5 and 6 | match |  |
| RQ3 summary box | token model vs. baseline AUC-ROC; Macro F1 vs. chance | 0.70; 0.78; 0.31 against 0.25 | 0.6984; 0.7848; 0.3124; 0.2455 | match |  |

## Summary boxes, Discussion, Threats, Conclusion

| Where | What | Paper | Reproduced | Status | Note |
|---|---|---|---|---|---|
| RQ1 summary box | supervised-assistant HR | HR = 0.76 | 0.7566 | match |  |
| RQ2 summary box | Cramer's V; corrective shares | V = 0.080; 35.6% vs. 32.5% | 0.07999; 35.577; 32.483 | match |  |
| Sec. 6.1 | difference largest after 90 days | HR lowest in the >90-day period | yes | match |  |
| Sec. 6.2 | supervised-assistant vs. human lines in test files (PR-weighted) | 35.6% vs. 24.1% | 35.618; 24.144 | match |  |
| Sec. 6.5 | Devin vs. human lines modified within two weeks | 12.0% vs. 9.6% | 11.952; 9.556 | match |  |
| Sec. 7 | agent made substantive changes in the mined cohort | 87.3% | 87.330 | match |  |
| Sec. 7 | second rater, Cohen's kappa | 100 PRs, kappa = 0.85 | 100; 0.8504 | match |  |
| Sec. 7 | merge commit not located | 359 merged PRs | 359 | match |  |
| Sec. 7 | Cursor and Claude Code contribute fewer than 50 PRs each | < 50 | 44, 31 | match |  |
| Sec. 7 | repositories | 215 repositories | 215 | match |  |
| Sec. 8 | related work | Xia and Miller: 182 repositories, HR = 1.11; Takerngsaksiri et al.: 3.68% vs. 2.34% within 30 days; Popescu et al.: three days, one week, three weeks; Kraishan: 90 days |  | external | literature |
| Sec. 9 | lines, PRs, repositories | 648,959 lines, 2,794 PRs, 215 repositories | 648,959; 2,794; 215 | match |  |
