# Analysis summary

Generated 2026-11-06 12:10 by `src/analysis/make_summary.py` from `expected_results/`.

**Primary design**: line level, full cohort (215 repositories, 2,794 PRs), vendored/generated paths excluded, every unit weighted 1/(units in its PR) with weights computed after the vendored filter, Cox model stratified by repository with robust SEs clustered by PR, covariates log(PR added lines) and log(PR source files). HR < 1 means agent-attributed code is modified later than human code.

## A. RQ1

### A1. Primary design and sensitivity variants

#### Line level

| Variant | Units | PRs | Repos | Agent HR (within-repo) | Supervised HR | Devin HR | Log-rank p |
|---|---|---|---|---|---|---|---|
| Primary (no vendored, PR-weighted) | 648,959 | 2,794 | 215 | 0.835 [0.749, 0.930], p=0.00107 | 0.757 [0.655, 0.875], p=0.000161 | 0.953 [0.820, 1.108], p=0.532 | 0.0317 |
| (a) all lines, PR-weighted (old primary) | 854,139 | 2,802 | 215 | 0.833 [0.748, 0.928], p=0.00086 | 0.758 [0.657, 0.875], p=0.000149 | 0.948 [0.817, 1.101], p=0.485 | 0.0287 |
| (b) no vendored, unweighted | 648,959 | 2,794 | 215 | 0.908 [0.687, 1.199], p=0.495 | 0.799 [0.595, 1.072], p=0.135 | 1.059 [0.655, 1.713], p=0.815 | 4.33e-267 |
| (c) all lines, unweighted | 854,139 | 2,802 | 215 | 0.923 [0.684, 1.246], p=0.601 | 0.863 [0.578, 1.289], p=0.473 | 0.999 [0.677, 1.476], p=0.998 | 0 |
| (d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted | 277,283 | 2,794 | 215 | 0.834 [0.749, 0.928], p=0.000924 | 0.760 [0.658, 0.878], p=0.00019 | 0.945 [0.815, 1.097], p=0.458 | 0.0284 |

Modification rates (unweighted / PR-weighted), PRs per group:

| Variant | Supervised assistants | Devin (autonomous) | Human | Agent (all) |
|---|---|---|---|---|
| Primary (no vendored, PR-weighted) | 23.34% / 24.13% (1,055 PRs, 179,124 units) | 23.64% / 37.95% (412 PRs, 87,793 units) | 30.10% / 34.51% (1,327 PRs, 382,042 units) | 23.44% / 28.01% (1,467 PRs, 266,917 units) |
| (a) all lines, PR-weighted (old primary) | 23.01% / 24.26% (1,058 PRs, 186,228 units) | 23.86% / 38.06% (412 PRs, 88,050 units) | 50.33% / 34.76% (1,332 PRs, 579,861 units) | 23.28% / 28.13% (1,470 PRs, 274,278 units) |
| (b) no vendored, unweighted | 23.34% / 24.13% (1,055 PRs, 179,124 units) | 23.64% / 37.95% (412 PRs, 87,793 units) | 30.10% / 34.51% (1,327 PRs, 382,042 units) | 23.44% / 28.01% (1,467 PRs, 266,917 units) |
| (c) all lines, unweighted | 23.01% / 24.26% (1,058 PRs, 186,228 units) | 23.86% / 38.06% (412 PRs, 88,050 units) | 50.33% / 34.76% (1,332 PRs, 579,861 units) | 23.28% / 28.13% (1,470 PRs, 274,278 units) |
| (d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted | 21.48% / 24.30% (1,055 PRs, 92,444 units) | 37.52% / 37.98% (412 PRs, 37,707 units) | 32.48% / 34.72% (1,327 PRs, 147,132 units) | 26.13% / 28.14% (1,467 PRs, 130,151 units) |

#### Hunk level

| Variant | Units | PRs | Repos | Agent HR (within-repo) | Supervised HR | Devin HR | Log-rank p |
|---|---|---|---|---|---|---|---|
| Primary (no vendored, PR-weighted) | 60,095 | 2,794 | 215 | 0.930 [0.849, 1.019], p=0.121 | 0.818 [0.725, 0.923], p=0.00109 | 1.113 [0.982, 1.262], p=0.0925 | 0.353 |
| (a) all lines, PR-weighted (old primary) | 62,529 | 2,802 | 215 | 0.930 [0.849, 1.018], p=0.118 | 0.820 [0.727, 0.924], p=0.00116 | 1.111 [0.980, 1.259], p=0.1 | 0.369 |
| (b) no vendored, unweighted | 60,095 | 2,794 | 215 | 1.060 [0.917, 1.226], p=0.429 | 0.927 [0.773, 1.112], p=0.415 | 1.236 [1.001, 1.527], p=0.0494 | 0.521 |
| (c) all lines, unweighted | 62,529 | 2,802 | 215 | 1.092 [0.938, 1.271], p=0.258 | 0.956 [0.794, 1.152], p=0.638 | 1.270 [1.017, 1.585], p=0.035 | 0.794 |
| (d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted | 42,631 | 2,794 | 215 | 0.926 [0.845, 1.014], p=0.0981 | 0.814 [0.722, 0.919], p=0.000867 | 1.106 [0.976, 1.254], p=0.113 | 0.314 |

Modification rates (unweighted / PR-weighted), PRs per group:

| Variant | Supervised assistants | Devin (autonomous) | Human | Agent (all) |
|---|---|---|---|---|
| Primary (no vendored, PR-weighted) | 30.91% / 34.45% (1,055 PRs, 17,927 units) | 51.10% / 52.51% (412 PRs, 5,499 units) | 39.30% / 44.77% (1,327 PRs, 36,669 units) | 35.65% / 39.52% (1,467 PRs, 23,426 units) |
| (a) all lines, PR-weighted (old primary) | 30.53% / 34.73% (1,058 PRs, 19,219 units) | 52.02% / 52.56% (412 PRs, 5,604 units) | 38.97% / 44.97% (1,332 PRs, 37,706 units) | 35.38% / 39.73% (1,470 PRs, 24,823 units) |
| (b) no vendored, unweighted | 30.91% / 34.45% (1,055 PRs, 17,927 units) | 51.10% / 52.51% (412 PRs, 5,499 units) | 39.30% / 44.77% (1,327 PRs, 36,669 units) | 35.65% / 39.52% (1,467 PRs, 23,426 units) |
| (c) all lines, unweighted | 30.53% / 34.73% (1,058 PRs, 19,219 units) | 52.02% / 52.56% (412 PRs, 5,604 units) | 38.97% / 44.97% (1,332 PRs, 37,706 units) | 35.38% / 39.73% (1,470 PRs, 24,823 units) |
| (d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted | 34.08% / 34.49% (1,055 PRs, 13,796 units) | 53.92% / 52.44% (412 PRs, 4,835 units) | 42.10% / 44.90% (1,327 PRs, 24,000 units) | 39.22% / 39.53% (1,467 PRs, 18,631 units) |

#### File level

| Variant | Units | PRs | Repos | Agent HR (within-repo) | Supervised HR | Devin HR | Log-rank p |
|---|---|---|---|---|---|---|---|
| Primary (no vendored, PR-weighted) | 18,120 | 2,794 | 215 | 0.970 [0.902, 1.043], p=0.412 | 0.947 [0.864, 1.039], p=0.25 | 1.008 [0.903, 1.125], p=0.883 | 0.49 |
| (a) all lines, PR-weighted (old primary) | 18,308 | 2,802 | 215 | 0.966 [0.899, 1.038], p=0.348 | 0.941 [0.859, 1.032], p=0.199 | 1.007 [0.902, 1.124], p=0.898 | 0.484 |
| (b) no vendored, unweighted | 18,120 | 2,794 | 215 | 0.894 [0.791, 1.010], p=0.0729 | 0.859 [0.729, 1.013], p=0.0706 | 0.947 [0.805, 1.114], p=0.509 | 2e-12 |
| (c) all lines, unweighted | 18,308 | 2,802 | 215 | 0.893 [0.791, 1.009], p=0.0692 | 0.858 [0.729, 1.011], p=0.0672 | 0.946 [0.804, 1.113], p=0.503 | 5e-14 |
| (d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted | 12,220 | 2,794 | 215 | 0.969 [0.901, 1.041], p=0.388 | 0.944 [0.861, 1.035], p=0.222 | 1.010 [0.904, 1.127], p=0.865 | 0.416 |

Modification rates (unweighted / PR-weighted), PRs per group:

| Variant | Supervised assistants | Devin (autonomous) | Human | Agent (all) |
|---|---|---|---|---|
| Primary (no vendored, PR-weighted) | 77.08% / 81.31% (1,055 PRs, 5,895 units) | 85.62% / 91.51% (412 PRs, 2,059 units) | 85.20% / 88.18% (1,327 PRs, 10,166 units) | 79.29% / 84.18% (1,467 PRs, 7,954 units) |
| (a) all lines, PR-weighted (old primary) | 77.02% / 81.39% (1,058 PRs, 6,028 units) | 85.64% / 91.51% (412 PRs, 2,061 units) | 85.25% / 88.22% (1,332 PRs, 10,219 units) | 79.22% / 84.22% (1,470 PRs, 8,089 units) |
| (b) no vendored, unweighted | 77.08% / 81.31% (1,055 PRs, 5,895 units) | 85.62% / 91.51% (412 PRs, 2,059 units) | 85.20% / 88.18% (1,327 PRs, 10,166 units) | 79.29% / 84.18% (1,467 PRs, 7,954 units) |
| (c) all lines, unweighted | 77.02% / 81.39% (1,058 PRs, 6,028 units) | 85.64% / 91.51% (412 PRs, 2,061 units) | 85.25% / 88.22% (1,332 PRs, 10,219 units) | 79.22% / 84.22% (1,470 PRs, 8,089 units) |
| (d) sampled design (20 files/PR, 100 lines/file, seed 42), no vendored, PR-weighted | 79.98% / 81.32% (1,055 PRs, 4,231 units) | 90.03% / 91.41% (412 PRs, 1,675 units) | 87.00% / 88.41% (1,327 PRs, 6,314 units) | 82.83% / 84.16% (1,467 PRs, 5,906 units) |

### A2. Pooled Cox, no strata, covariates log PR added lines, files, stars, contributors, robust SEs clustered by PR

| Level | Primary | All units, unweighted |
|---|---|---|
| line | 0.894 [0.803, 0.996], p=0.0424 | 0.755 [0.520, 1.096], p=0.139 |
| hunk | 0.985 [0.898, 1.079], p=0.739 | 0.965 [0.782, 1.190], p=0.74 |
| file | 0.988 [0.921, 1.059], p=0.731 | 0.848 [0.724, 0.994], p=0.0415 |
### A3. RMST and modification-rate differences (primary design, repository cluster bootstrap)

| Level | B | Mod. rate agent | Mod. rate human | Diff (pp) [95% CI] | RMST90 agent / human (days) | RMST90 diff [CI] | RMST150 agent / human | RMST150 diff [CI] |
|---|---|---|---|---|---|---|---|---|
| line | 1000 (1000 ok) | 28.0091% | 34.5121% | -6.5030 [-10.6789, -1.6546] | 76.730 / 75.954 | 0.777 [-2.311, 3.826] | 122.649 / 120.455 | 2.194 [-3.443, 7.872] |
| hunk | 1000 (1000 ok) | 39.5232% | 44.7712% | -5.2480 [-9.6463, -0.3522] | 70.348 / 70.615 | -0.267 [-3.253, 3.051] | 110.283 / 109.882 | 0.401 [-5.207, 6.536] |
| file | 1000 (1000 ok) | 84.1770% | 88.1833% | -4.0063 [-7.0849, -0.8459] | 33.943 / 33.724 | 0.219 [-3.303, 3.725] | 46.563 / 45.288 | 1.275 [-3.847, 6.382] |

Modification rates in A3 are PR-weighted (each PR counts once) and unrounded.

### A4. Proportional hazards (primary line model)

Schoenfeld-residual test (lifelines `proportional_hazard_test`, rank transform):

- agent_weighted_robust: is_agent stat=481278.52 p=0; n_added_lines stat=623657.82 p=0; n_source_files stat=1195768.80 p=0
- agent_unweighted: is_agent stat=790.81 p=5.36e-174; n_added_lines stat=626.97 p=2.28e-138; n_source_files stat=101.75 p=6.31e-24
- tier_weighted_robust: devin stat=1140079.47 p=0; n_added_lines stat=642955.25 p=0; n_source_files stat=1249381.75 p=0; supervised stat=35756.98 p=0
- tier_unweighted: devin stat=2090.73 p=0; n_added_lines stat=632.85 p=1.2e-139; n_source_files stat=219.45 p=1.19e-49; supervised stat=12.68 p=0.000369

The weighted test statistics are not usable: lifelines treats the fractional PR weights as frequency weights in the residual test, which inflates the statistics to 10^5-10^6. Use the unweighted model test (same stratified model without weights and clustering). With 650k lines, any departure from PH is "significant"; the time-split HRs below show its size and direction.

Time-split within-repo HRs (primary design; separate stratified Cox fits on the units at risk at the start of each period):

| Period (days) | Agent | Supervised | Devin |
|---|---|---|---|
| 0-14 | 1.009 [0.830, 1.226], p=0.93 | 1.101 [0.840, 1.441], p=0.486 | 0.907 [0.695, 1.182], p=0.469 |
| 14-90 | 0.853 [0.720, 1.009], p=0.064 | 0.794 [0.638, 0.987], p=0.0379 | 0.942 [0.740, 1.199], p=0.626 |
| 90-end | 0.652 [0.532, 0.799], p=3.74e-05 | 0.471 [0.363, 0.611], p=1.44e-08 | 1.027 [0.786, 1.341], p=0.846 |

### A5. Per-agent results (primary design)

#### Line level

| Agent | Units | PRs | Repos | Within-repo HR vs Human | PR-weighted mod. rate [PR bootstrap 95% CI] | Unweighted rate | Share of units from largest PR (PR) |
|---|---|---|---|---|---|---|---|
| Copilot | 67,440 | 489 | 60 | 0.746 [0.595, 0.936], p=0.0114 | 22.45% [19.62%, 25.28%] | 15.82% | 9.69% ('microsoft/typespec', np.int64(7927)) |
| OpenAI_Codex | 50,081 | 491 | 75 | 0.693 [0.570, 0.843], p=0.000244 | 24.41% [21.42%, 27.82%] | 25.06% | 5.66% ('block/goose', np.int64(2621)) |
| Cursor | 13,343 | 44 | 18 | 1.807 [0.907, 3.602], p=0.0925 | 35.04% [24.60%, 46.65%] | 34.30% | 26.40% ('risingwavelabs/risingwave', np.int64(22554)) |
| Claude_Code | 48,260 | 31 | 14 | 1.664 [0.813, 3.407], p=0.164 | 30.48% [18.00%, 43.53%] | 29.03% | 32.87% ('awslabs/mcp', np.int64(427)) |
| Devin | 87,793 | 412 | 31 | 0.957 [0.823, 1.112], p=0.564 | 37.95% [34.38%, 41.46%] | 23.64% | 41.62% ('pdfme/pdfme', np.int64(1059)) |
| Human | 382,042 | 1327 | 193 | reference | 34.51% [32.61%, 36.58%] | 30.10% | 9.00% ('gridaco/grida', np.int64(383)) |

#### Hunk level

| Agent | Units | PRs | Repos | Within-repo HR vs Human | PR-weighted mod. rate [PR bootstrap 95% CI] | Unweighted rate | Share of units from largest PR (PR) |
|---|---|---|---|---|---|---|---|
| Copilot | 8,723 | 489 | 60 | 0.822 [0.682, 0.990], p=0.0389 | 33.19% [30.05%, 36.49%] | 24.66% | 5.64% ('azure/azure-sdk-for-js', np.int64(34737)) |
| OpenAI_Codex | 7,062 | 491 | 75 | 0.754 [0.638, 0.890], p=0.000857 | 33.56% [30.13%, 36.99%] | 33.21% | 8.74% ('logancyang/obsidian-copilot', np.int64(1497)) |
| Cursor | 726 | 44 | 18 | 1.213 [0.653, 2.251], p=0.541 | 43.75% [31.96%, 55.50%] | 41.46% | 22.04% ('latitude-dev/latitude-llm', np.int64(1390)) |
| Claude_Code | 1,416 | 31 | 14 | 1.593 [0.924, 2.747], p=0.094 | 55.36% [40.97%, 70.48%] | 52.61% | 29.94% ('coder/coder', np.int64(17035)) |
| Devin | 5,499 | 412 | 31 | 1.115 [0.984, 1.264], p=0.0872 | 52.51% [48.98%, 56.04%] | 51.10% | 4.29% ('pdfme/pdfme', np.int64(1059)) |
| Human | 36,669 | 1327 | 193 | reference | 44.77% [42.84%, 46.74%] | 39.30% | 5.23% ('tinyplex/tinybase', np.int64(227)) |

#### File level

| Agent | Units | PRs | Repos | Within-repo HR vs Human | PR-weighted mod. rate [PR bootstrap 95% CI] | Unweighted rate | Share of units from largest PR (PR) |
|---|---|---|---|---|---|---|---|
| Copilot | 2,747 | 489 | 60 | 0.873 [0.764, 0.999], p=0.0475 | 77.23% [74.09%, 80.08%] | 68.66% | 7.24% ('azure/azure-sdk-for-js', np.int64(34737)) |
| OpenAI_Codex | 2,374 | 491 | 75 | 1.024 [0.896, 1.170], p=0.73 | 84.94% [82.15%, 87.67%] | 84.33% | 7.75% ('prebid/prebid.js', np.int64(13288)) |
| Cursor | 281 | 44 | 18 | 0.853 [0.569, 1.278], p=0.441 | 81.96% [72.19%, 90.95%] | 88.26% | 28.11% ('latitude-dev/latitude-llm', np.int64(1390)) |
| Claude_Code | 493 | 31 | 14 | 0.979 [0.684, 1.403], p=0.909 | 87.30% [78.33%, 95.66%] | 82.76% | 35.90% ('coder/coder', np.int64(17035)) |
| Devin | 2,059 | 412 | 31 | 1.013 [0.908, 1.131], p=0.811 | 91.51% [89.32%, 93.74%] | 85.62% | 11.46% ('pdfme/pdfme', np.int64(1059)) |
| Human | 10,166 | 1327 | 193 | reference | 88.18% [86.80%, 89.62%] | 85.20% | 7.15% ('azure/azure-sdk-for-js', np.int64(32546)) |

### A6. 14-day modification rate (primary design: non-vendored, PR-weighted)

| Level | Supervised assistants | Devin (autonomous) | Human | Agent (all) |
|---|---|---|---|---|
| line | 8.30% (unweighted 10.84%) | 11.95% (unweighted 11.38%) | 9.56% (unweighted 8.68%) | 9.32% (unweighted 11.02%) |
| hunk | 12.13% (unweighted 9.43%) | 19.58% (unweighted 24.60%) | 13.65% (unweighted 11.68%) | 14.22% (unweighted 12.99%) |
| file | 44.55% (unweighted 34.64%) | 56.96% (unweighted 54.88%) | 47.75% (unweighted 48.04%) | 48.04% (unweighted 39.88%) |

### A7. Substantial modification (theta = share of the unit's PR-added lines modified), primary design

| Unit, theta | Units | Mod. rate PR-weighted (Sup / Devin / Human) | Agent HR | Supervised HR | Devin HR |
|---|---|---|---|---|---|
| hunk|25% | 60,095 | 28.17% / 43.74% / 39.13% | 0.860 [0.778, 0.951], p=0.00323 | 0.764 [0.669, 0.874], p=8.28e-05 | 1.009 [0.878, 1.160], p=0.9 |
| hunk|50% | 60,095 | 25.56% / 39.81% / 36.38% | 0.836 [0.752, 0.928], p=0.000798 | 0.761 [0.662, 0.876], p=0.000138 | 0.947 [0.818, 1.096], p=0.465 |
| hunk|75% | 60,095 | 23.29% / 36.22% / 33.75% | 0.815 [0.731, 0.910], p=0.000275 | 0.732 [0.632, 0.847], p=2.92e-05 | 0.946 [0.811, 1.104], p=0.483 |
| hunk|100% | 60,095 | 22.54% / 34.67% / 32.68% | 0.808 [0.723, 0.903], p=0.000176 | 0.725 [0.625, 0.840], p=1.93e-05 | 0.940 [0.805, 1.099], p=0.439 |
| file|25% | 18,120 | 29.36% / 47.47% / 41.94% | 0.851 [0.768, 0.944], p=0.00227 | 0.749 [0.652, 0.860], p=4.45e-05 | 1.009 [0.874, 1.166], p=0.898 |
| file|50% | 18,120 | 24.30% / 38.33% / 34.51% | 0.831 [0.742, 0.930], p=0.00125 | 0.765 [0.657, 0.891], p=0.000556 | 0.928 [0.792, 1.086], p=0.351 |
| file|75% | 18,120 | 20.07% / 31.66% / 29.14% | 0.804 [0.709, 0.911], p=0.000615 | 0.719 [0.610, 0.848], p=8.61e-05 | 0.936 [0.783, 1.119], p=0.466 |
| file|100% | 18,120 | 17.68% / 27.93% / 24.84% | 0.849 [0.742, 0.972], p=0.0177 | 0.753 [0.632, 0.897], p=0.00149 | 1.003 [0.828, 1.216], p=0.974 |

### A8. Attribution audit vs. the mined cohort

Of the 325 audited PRs (drawn from the 2,092 agent PRs with a located merge commit), **221 are in the mined cohort** (2,802 PRs), and 221 in the primary design. The other 104 PRs add no tracked source line.

| Scope | n | agent | mixed | human | unclear | agent or mixed |
|---|---|---|---|---|---|---|
| mined|all | 221 | 136 (61.54% [54.98%, 67.70%]) | 57 (25.79% [20.47%, 31.94%]) | 16 (7.24% [4.51%, 11.44%]) | 12 (5.43% [3.13%, 9.25%]) | 193 (87.33% [82.30%, 91.09%]) |
| mined|Supervised assistants | 157 | 92 (58.60% [50.78%, 66.01%]) | 38 (24.20% [18.17%, 31.47%]) | 16 (10.19% [6.37%, 15.91%]) | 11 (7.01% [3.96%, 12.11%]) | 130 (82.80% [76.13%, 87.90%]) |
| mined|Devin (autonomous) | 64 | 44 (68.75% [56.61%, 78.77%]) | 19 (29.69% [19.91%, 41.77%]) | 0 (0.00% [0.00%, 5.66%]) | 1 (1.56% [0.28%, 8.33%]) | 63 (98.44% [91.67%, 99.72%]) |

Labels of mined audited PRs by agent: {'Claude_Code': {'human': 4, 'agent': 1}, 'Copilot': {'agent': 36, 'mixed': 32, 'human': 2}, 'Cursor': {'agent': 5, 'unclear': 1}, 'Devin': {'agent': 44, 'mixed': 19, 'unclear': 1}, 'OpenAI_Codex': {'agent': 50, 'unclear': 10, 'human': 10, 'mixed': 6}}. Intervals are 95% Wilson score intervals.

Primary RQ1 (line) under attribution restrictions:

| Restriction | Units | PRs | Agent PRs | Agent HR | Supervised HR | Devin HR |
|---|---|---|---|---|---|---|
| excl_human | 631,276 | 2,778 | 1451 | 0.835 [0.749, 0.931], p=0.00116 | 0.757 [0.654, 0.875], p=0.000174 | 0.953 [0.820, 1.107], p=0.529 |
| excl_human_unclear | 625,611 | 2,766 | 1439 | 0.831 [0.745, 0.927], p=0.000868 | 0.748 [0.646, 0.867], p=0.000109 | 0.954 [0.821, 1.108], p=0.536 |
| agent_label_only_vs_all_human | 394,699 | 1,463 | 136 | 0.889 [0.640, 1.233], p=0.48 | 1.032 [0.660, 1.613], p=0.89 | 0.728 [0.452, 1.172], p=0.191 |
| agent_or_mixed_label_vs_all_human | 442,643 | 1,520 | 193 | 0.821 [0.636, 1.062], p=0.133 | 0.850 [0.583, 1.239], p=0.398 | 0.793 [0.562, 1.119], p=0.187 |

### A9. File-level covariates at merge time (primary design)

Covariates per unit's file, read from local clones by `src/mining/a9_extract.py` (shipped in `data/file_covariates/`): prior 3-month modification count (commits reachable from the PR base touching the path in the 90 days before the merge; the PR itself excluded), file age in days (first commit adding the path; 0 for new files), new file (path absent at the PR base), test path (directory test/tests/spec/specs/__tests__ or "test"/"spec" in the file name). Counts and age enter as log1p.

| Level | Units with features | Agent HR, same rows, base model | Agent HR + file covariates | Supervised base -> +cov | Devin base -> +cov |
|---|---|---|---|---|---|
| line | 648,746 / 648,959 | 0.833 [0.748, 0.928], p=0.000941 | 0.879 [0.789, 0.979], p=0.0188 | 0.755 [0.653, 0.872], p=0.000137 -> 0.807 [0.699, 0.931], p=0.00336 | 0.952 [0.820, 1.106], p=0.524 -> 0.984 [0.846, 1.145], p=0.835 |
| hunk | 60,052 / 60,095 | 0.930 [0.849, 1.018], p=0.117 | 0.939 [0.857, 1.029], p=0.175 | 0.817 [0.724, 0.922], p=0.00104 -> 0.843 [0.748, 0.950], p=0.00525 | 1.113 [0.982, 1.261], p=0.0947 -> 1.088 [0.956, 1.238], p=0.202 |
| file | 18,106 / 18,120 | 0.972 [0.904, 1.045], p=0.44 | 1.040 [0.968, 1.117], p=0.285 | 0.950 [0.866, 1.042], p=0.278 -> 1.019 [0.933, 1.113], p=0.671 | 1.008 [0.903, 1.125], p=0.886 -> 1.073 [0.958, 1.202], p=0.223 |

Covariate HRs (agent model, line level): is_agent 0.879 [0.789, 0.979], p=0.0188; prior_mod_3m 1.261 [1.192, 1.334], p=7.14e-16; file_age_days 0.913 [0.880, 0.948], p=2.04e-06; new_file 0.833 [0.667, 1.041], p=0.108; test 0.729 [0.642, 0.828], p=1.09e-06

PR-weighted descriptives (line level): Devin (autonomous): new file 24.57%, test path 26.82%, median prior 3-month changes 0, median file age 0 d; Human: new file 15.75%, test path 24.14%, median prior 3-month changes 0, median file age 0 d; Supervised assistants: new file 15.45%, test path 35.62%, median prior 3-month changes 0, median file age 0 d; Agent (all): new file 18.02%, test path 33.14%, median prior 3-month changes 0, median file age 0 d

### A10. Repository-activity terciles (primary line design)

Commit counts on the default branch are not stored for the cohort; the two activity measures are those of the Q1 filter: AIDev PR count of the repository (agent + human, all states, `data/repo_meta_v6.csv: total_pr_count`) and open issues. Terciles by rank.

| Measure, tercile (range, repos) | Units | PRs | Agent HR | Supervised HR | Devin HR |
|---|---|---|---|---|---|
| total_pr_count|low (2-8, 72) | 34,718 | 158 | 0.832 [0.440, 1.572], p=0.57 | 0.821 [0.425, 1.588], p=0.558 | 1.113 [0.214, 5.794], p=0.899 |
| total_pr_count|mid (8-20, 71) | 65,899 | 322 | 0.634 [0.428, 0.939], p=0.0231 | 0.627 [0.420, 0.937], p=0.0227 | 0.692 [0.238, 2.009], p=0.498 |
| total_pr_count|high (20-620, 72) | 548,342 | 2,314 | 0.863 [0.770, 0.967], p=0.0113 | 0.783 [0.668, 0.919], p=0.00266 | 0.966 [0.830, 1.124], p=0.655 |
| open_issues|low (0-136, 72) | 224,213 | 746 | 0.782 [0.623, 0.982], p=0.0345 | 0.609 [0.450, 0.823], p=0.00126 | 0.984 [0.726, 1.334], p=0.919 |
| open_issues|mid (137-335, 71) | 206,541 | 947 | 0.879 [0.725, 1.064], p=0.185 | 0.762 [0.583, 0.995], p=0.0456 | 1.045 [0.810, 1.350], p=0.733 |
| open_issues|high (338-12345, 72) | 218,205 | 1,101 | 0.836 [0.713, 0.981], p=0.0284 | 0.809 [0.659, 0.995], p=0.0443 | 0.885 [0.701, 1.118], p=0.306 |

The Q1-removed repositories themselves were never mined, so a "keep Q1" sensitivity is not possible without new mining.

### A11. Descriptives per group

| Group | All lines | Share of all lines in vendored paths | Non-vendored lines | Share in test paths (unweighted / PR-weighted) | PRs | Merge date Q1 / median / Q3 (min-max) |
|---|---|---|---|---|---|---|
| Devin (autonomous) | 88,050 | 0.29% | 87,793 | 28.92% / 25.17% | 412 | 2025-02-24 / 2025-04-09 / 2025-06-01 (2024-12-28 to 2025-07-30) |
| Human | 579,861 | 34.11% | 382,042 | 20.77% / 23.31% | 1327 | 2025-02-22 / 2025-04-09 / 2025-05-17 (2025-01-01 to 2025-06-28) |
| Supervised assistants | 186,228 | 3.81% | 179,124 | 42.83% / 34.86% | 1055 | 2025-06-04 / 2025-06-18 / 2025-07-08 (2025-03-26 to 2025-07-30) |
| Agent (all) | 274,278 | 2.68% | 266,917 | 38.26% / 32.14% | 1467 | 2025-05-21 / 2025-06-11 / 2025-07-03 (2024-12-28 to 2025-07-30) |

Merged PRs per month: Devin (autonomous): {'2024-12': 6, '2025-01': 48, '2025-02': 55, '2025-03': 64, '2025-04': 71, '2025-05': 64, '2025-06': 42, '2025-07': 62}; Human: {'2025-01': 174, '2025-02': 215, '2025-03': 208, '2025-04': 255, '2025-05': 251, '2025-06': 224}; Supervised assistants: {'2025-03': 1, '2025-04': 1, '2025-05': 212, '2025-06': 480, '2025-07': 361}

Test paths here: directory test/tests/spec/__tests__ or "test" in the file name (A9 also counts "spec" in the name). New-file shares are in A9.

### A12. Table 3 for the primary design

| Level | Group | Units | PRs | % modified unweighted | % modified PR-weighted | Median days to modification (unweighted / PR-weighted) | Median follow-up of censored units (days) |
|---|---|---|---|---|---|---|---|
| line | Devin (autonomous) | 87,793 | 412 | 23.64% | 37.95% | 17.61 / 42.42 | 188.6 |
| line | Human | 382,042 | 1,327 | 30.10% | 34.51% | 41.90 / 48.02 | 260.3 |
| line | Supervised assistants | 179,124 | 1,055 | 23.34% | 24.13% | 14.45 / 30.91 | 195.6 |
| line | Agent (all) | 266,917 | 1,467 | 23.44% | 28.01% | 14.81 / 36.03 | 191.3 |
| line | All | 648,959 | 2,794 | 27.36% | 31.10% | 38.99 / 40.13 | 218.3 |
| hunk | Devin (autonomous) | 5,499 | 412 | 51.10% | 52.51% | 17.06 / 34.96 | 267.2 |
| hunk | Human | 36,669 | 1,327 | 39.30% | 44.77% | 44.42 / 41.40 | 270.1 |
| hunk | Supervised assistants | 17,927 | 1,055 | 30.91% | 34.45% | 36.81 / 29.47 | 194.3 |
| hunk | Agent (all) | 23,426 | 1,467 | 35.65% | 39.52% | 29.52 / 32.06 | 198.0 |
| hunk | All | 60,095 | 2,794 | 37.88% | 42.02% | 40.07 / 35.89 | 236.5 |
| file | Devin (autonomous) | 2,059 | 412 | 85.62% | 91.51% | 5.87 / 6.28 | 188.6 |
| file | Human | 10,166 | 1,327 | 85.20% | 88.18% | 10.07 / 10.93 | 258.4 |
| file | Supervised assistants | 5,895 | 1,055 | 77.08% | 81.31% | 18.43 / 11.03 | 186.5 |
| file | Agent (all) | 7,954 | 1,467 | 79.29% | 84.18% | 13.88 / 9.45 | 188.6 |
| file | All | 18,120 | 2,794 | 82.60% | 86.08% | 11.57 / 9.99 | 216.5 |

### A13. KM figure

`figures/rq1_km_line_primary.pdf` (and .png): PR-weighted KM curves, primary line data, no title, legend colours match the lines, untruncated group names in the at-risk table.

Lines at risk: Human: {'0': 382042, '60': 316726, '120': 289549, '180': 278814, '240': 179051, '300': 48384, '360': 1685}; Supervised assistants: {'0': 179124, '60': 149526, '120': 144209, '180': 91356, '240': 918, '300': 0, '360': 0}; Devin (autonomous): {'0': 87793, '60': 74322, '120': 70754, '180': 66568, '240': 20350, '300': 8497, '360': 680}
- Human: last unit at risk at day 364.2. Largest PR-weighted step at day 338.4 (drop 0.0020), all from PR ('nethermindeth/nethermind', np.int64(8038)). Largest unweighted step at day 322.6 (drop 0.0152), mostly PR ('gridaco/grida', np.int64(218)) (573 lines).
- Supervised assistants: last unit at risk at day 280.7. Largest PR-weighted step at day 223.4 (drop 0.0116), all from PR ('elsa-workflows/elsa-core', np.int64(6654)). Largest unweighted step at day 1.0 (drop 0.0355), mostly PR ('awslabs/mcp', np.int64(427)) (6358 lines).
- Devin (autonomous): last unit at risk at day 369.0. Largest PR-weighted step at day 313.1 (drop 0.0119), all from PR ('promptfoo/promptfoo', np.int64(3187)). Largest unweighted step at day 304.9 (drop 0.0564), mostly PR ('crewaiinc/crewai', np.int64(1828)) (573 lines).

Follow-up ends on 2025-12-31, so groups whose PRs were merged later (A11) have no unit at risk at the longest durations.


## C. RQ2 (primary units: non-vendored lines; word-boundary classifier; one observation per modifying commit x author group)

All lines (with vendored paths), word classifier (all lines, word, commit x group): n=6,182, chi2=42.57, V=0.0830 .

**Primary**: n=6,112 (agent 2,347, human 3,765), chi2=39.10, dof=4, p=6.63e-08, Cramer's V=0.0800

| Class | Agent % | Human % | Std. residual agent | Std. residual human |
|---|---|---|---|---|
| Adaptive | 11.67 | 12.72 | -0.89 | +0.70 |
| Corrective | 35.58 | 32.48 | +1.59 | -1.26 |
| Other | 10.57 | 15.91 | -4.28 | +3.38 |
| Perfective | 38.30 | 35.35 | +1.46 | -1.15 |
| Preventive | 3.88 | 3.53 | +0.54 | -0.42 |

**Excluding merge commits**: n=5,656 (agent 2,206, human 3,450), chi2=37.11, dof=4, p=1.71e-07, Cramer's V=0.0810

| Class | Agent % | Human % | Std. residual agent | Std. residual human |
|---|---|---|---|---|
| Adaptive | 8.30 | 8.41 | -0.11 | +0.09 |
| Corrective | 36.63 | 33.91 | +1.31 | -1.05 |
| Other | 11.24 | 17.07 | -4.34 | +3.47 |
| Perfective | 39.76 | 36.84 | +1.35 | -1.08 |
| Preventive | 4.08 | 3.77 | +0.45 | -0.36 |

**Only commits touching one group**: n=5,088 (agent 1,835, human 3,253), chi2=35.83, dof=4, p=3.13e-07, Cramer's V=0.0839

| Class | Agent % | Human % | Std. residual agent | Std. residual human |
|---|---|---|---|---|
| Adaptive | 12.15 | 13.16 | -0.77 | +0.58 |
| Corrective | 36.24 | 32.37 | +1.82 | -1.37 |
| Other | 11.28 | 17.15 | -4.15 | +3.12 |
| Perfective | 36.51 | 33.88 | +1.22 | -0.92 |
| Preventive | 3.81 | 3.44 | +0.54 | -0.40 |

**Substring classifier**: n=6,112 (agent 2,347, human 3,765), chi2=146.19, dof=4, p=1.33e-30, Cramer's V=0.1547

| Class | Agent % | Human % | Std. residual agent | Std. residual human |
|---|---|---|---|---|
| Adaptive | 7.37 | 9.80 | -2.43 | +1.92 |
| Corrective | 28.93 | 31.50 | -1.39 | +1.10 |
| Other | 4.05 | 9.51 | -5.99 | +4.73 |
| Perfective | 29.91 | 30.39 | -0.26 | +0.20 |
| Preventive | 29.74 | 18.80 | +6.80 | -5.37 |

**File x commit unit (non-vendored files), word**: n=5,832 (agent 2,681, human 3,151), chi2=7.02, dof=4, p=0.135, Cramer's V=0.0347

| Class | Agent % | Human % | Std. residual agent | Std. residual human |
|---|---|---|---|---|
| Adaptive | 9.14 | 9.84 | -0.63 | +0.59 |
| Corrective | 39.09 | 37.48 | +0.73 | -0.67 |
| Other | 9.85 | 11.71 | -1.58 | +1.46 |
| Perfective | 37.30 | 36.75 | +0.25 | -0.23 |
| Preventive | 4.63 | 4.22 | +0.54 | -0.50 |

**File x commit unit, substring**: n=5,832 (agent 2,681, human 3,151), chi2=100.67, dof=4, p=7.08e-21, Cramer's V=0.1314

| Class | Agent % | Human % | Std. residual agent | Std. residual human |
|---|---|---|---|---|
| Adaptive | 4.70 | 6.85 | -2.49 | +2.30 |
| Corrective | 31.56 | 35.07 | -1.70 | +1.57 |
| Other | 3.73 | 4.92 | -1.59 | +1.47 |
| Perfective | 25.74 | 30.37 | -2.44 | +2.25 |
| Preventive | 34.28 | 22.79 | +6.07 | -5.60 |

Merge commits (more than one parent, from data/rq2/commit_parents.csv): 416 distinct modifying commits, 456 of 6112 units (by group {'Human': 315, 'Agent': 141}; classified as {'Adaptive': 280, 'Perfective': 82, 'Corrective': 80, 'Other': 10, 'Preventive': 4}). Commits touching both groups: 512. Word vs substring agreement over all commits: 66.75%.

## D. RQ3 (vendored/generated files excluded from all inputs)

```
rq3a_all: events 18308 -> 18120 (vendored dropped 188); code rows 18308 -> 18120
rq3a_agent: events 8089 -> 7954 (vendored dropped 135); code rows 8089 -> 7954
rq3b_all: rows 15075 -> 14923 (vendored dropped 152); prior==0 old 275 (PR itself not counted? check) new 567; bins {0: 0.1563, 1: 0.245, 2: 0.2739, 3: 0.3248}
rq3b_agent: rows 6401 -> 6300 (vendored dropped 101); prior==0 old 97 (PR itself not counted? check) new 171; bins {0: 0.151, 1: 0.2502, 2: 0.2671, 3: 0.3317}
```

### D1. RQ3a, agent files (full 8-model 30x10 tournament)

| Model | AUC-PR mean [CI] | AUC-ROC mean [CI] | F1 | MCC |
|---|---|---|---|---|
| RandomForest | 0.8895 [0.7561, 0.9769] | 0.6984 [0.5295, 0.8202] | 0.6995 | 0.2192 |
| CatBoost | 0.8883 [0.7564, 0.9761] | 0.6955 [0.5390, 0.8128] | 0.7643 | 0.2260 |
| XGBoost | 0.8860 [0.7501, 0.9759] | 0.6936 [0.5259, 0.8152] | 0.7246 | 0.2244 |
| MLP | 0.8748 [0.7419, 0.9684] | 0.6624 [0.5476, 0.7801] | 0.8185 | 0.1622 |
| SVM | 0.8741 [0.7409, 0.9673] | 0.6405 [0.5258, 0.7536] | 0.7216 | 0.1797 |
| LogisticRegression | 0.8699 [0.7430, 0.9673] | 0.6333 [0.5200, 0.7357] | 0.7518 | 0.1544 |
| KNN | 0.8497 [0.6949, 0.9584] | 0.6361 [0.5455, 0.7208] | 0.5728 | 0.1526 |
| NaiveBayes | 0.8206 [0.6277, 0.9512] | 0.5144 [0.3790, 0.6496] | 0.6066 | 0.0310 |

Scott-Knott ESD (auc_pr): RandomForest (rank 1), CatBoost (rank 1), XGBoost (rank 1), MLP (rank 2), SVM (rank 2), LogisticRegression (rank 2), KNN (rank 3), NaiveBayes (rank 4)

Scott-Knott ESD (auc_roc): RandomForest (rank 1), CatBoost (rank 1), XGBoost (rank 1), MLP (rank 2), SVM (rank 3), KNN (rank 3), LogisticRegression (rank 3), NaiveBayes (rank 4)

LR baseline (log file lines + log prior 3-month changes, same 30x10 grouped CV): {"n": 6661, "pos_rate": 0.7975, "prior_missing": 0, "folds": 300, "auc_roc_mean": 0.7848, "auc_roc_median": 0.7903, "auc_roc_std": 0.0637, "auc_pr_mean": 0.9292, "auc_pr_median": 0.9382, "auc_pr_std": 0.0453, "auc_roc_p2_5": 0.6394, "auc_roc_p97_5": 0.8881, "auc_pr_p2_5": 0.8281, "auc_pr_p97_5": 0.9867}

Out-of-sample explanation: {"best_model": "RandomForest", "shap_model": "RandomForest", "n": 6661, "repos": 155, "pos_rate": 0.7975, "oos_auc_roc": 0.7043, "oos_auc_pr": 0.8977, "oos_f1": 0.6981, "shap_model_oos_auc_pr": 0.8977, "lime_tokens_in_shap_top50": 0.469, "shap_top_tokens": [{"token": "key", "mean_abs_shap": 0.01594, "mean_shap": 0.00241}, {"token": "model", "mean_abs_shap": 0.01179, "mean_shap": 0.00097}, {"token": "config", "mean_abs_shap": 0.01081, "mean_shap": 0.0012}, {"token": "error", "mean_abs_shap": 0.01036, "mean_shap": 0.00125}, {"token": "state", "mean_abs_shap": 0.00888, "mean_shap": 0.00115}, {"token": "url", "mean_abs_shap": 0.00752, "mean_shap": 0.00118}, {"token": "path", "mean_abs_shap": 0.00651, "mean_shap": 0.0011}, {"token": "prompt", "mean_abs_shap": 0.00644, "mean_shap": 0.00096}, {"token": "result", "mean_abs_shap": 0.00604, "mean_shap": 0.00075}, {"token": "json", "mean_abs_shap": 0.0059, "mean_shap": 0.00067}, {"token": "all", "mean_abs_shap": 0.0052, "mean_shap": 0.00062}, {"token": "name", "mean_abs_shap": 0.00494, "mean_shap": 0.00033}, {"token": "max", "mean_abs_shap": 0.00438, "mean_shap": 0.00048}, {"token": "update", "mean_abs_shap": 0.00419, "mean_shap": 0.00059}, {"token": "str", "mean_abs_shap": 0.00379, "mean_shap": 0.0003}, {"token": "api", "mean_abs_shap": 0.00368, "mean_shap": 0.00023}, {"token": "cache", "mean_abs_shap": 0.00364, "mean_shap": 0.00052}, {"token": "user", "mean_abs_shap": 0.0034, "mean_shap": 0.00048}, {"token": "schema", "mean_abs_shap": 0.00338, "mean_shap": 0.00037}, {"token": "description", "mean_abs_shap": 0.00318, "mean_shap": 0.00083}]}

Line localization: {"model": "RandomForest", "files": 1941, "repos": 109, "lines": 163409, "line_mod_rate": 0.2278, "pooled_auc_roc": 0.5465, "pooled_auc_pr": 0.2598, "length_baseline_pooled_auc_roc": 0.5523, "mean_file_auc_roc": 0.5758, "median_file_auc_roc": 0.5818, "precision_at_10pct": 0.3666, "precision_at_10pct_length_baseline": 0.3973, "base_rate_mean_per_file": 0.3006}

LIME runtime: {"calls": 5, "explanations": 100, "seconds_total": 20.9844, "seconds_per_explanation": 0.2098}

### D1. RQ3a, all files (comparison only: RandomForest, CatBoost, XGBoost + LR baseline, 30x10)

| Model | AUC-PR mean [CI] | AUC-ROC mean [CI] | F1 | MCC |
|---|---|---|---|---|
| RandomForest | 0.9119 [0.8156, 0.9779] | 0.6973 [0.5854, 0.8048] | 0.7066 | 0.2120 |
| XGBoost | 0.9096 [0.8096, 0.9751] | 0.6938 [0.5906, 0.7960] | 0.6816 | 0.2077 |
| CatBoost | 0.9091 [0.8079, 0.9777] | 0.6930 [0.5856, 0.8078] | 0.7836 | 0.2159 |

Scott-Knott ESD (auc_pr): RandomForest (rank 1), XGBoost (rank 1), CatBoost (rank 1)

Scott-Knott ESD (auc_roc): RandomForest (rank 1), XGBoost (rank 1), CatBoost (rank 1)

LR baseline (log file lines + log prior 3-month changes, same 30x10 grouped CV): {"n": 15946, "pos_rate": 0.838, "prior_missing": 0, "folds": 300, "auc_roc_mean": 0.7491, "auc_roc_median": 0.7565, "auc_roc_std": 0.0753, "auc_pr_mean": 0.9346, "auc_pr_median": 0.9397, "auc_pr_std": 0.0368, "auc_roc_p2_5": 0.5823, "auc_roc_p97_5": 0.8625, "auc_pr_p2_5": 0.8524, "auc_pr_p97_5": 0.9874}

Out-of-sample explanation: PENDING

Line localization: {"model": "RandomForest", "files": 4851, "repos": 153, "lines": 373641, "line_mod_rate": 0.2558, "pooled_auc_roc": 0.5422, "pooled_auc_pr": 0.288, "length_baseline_pooled_auc_roc": 0.5641, "mean_file_auc_roc": 0.5566, "median_file_auc_roc": 0.5667, "precision_at_10pct": 0.3848, "precision_at_10pct_length_baseline": 0.4163, "base_rate_mean_per_file": 0.322}

LIME runtime: PENDING

### D2. RQ3b, PR-history leakage fix

PR-history features (full leakage fix, `src/mining/extract_rq3_pr_features_fixed.py`): prior = author PRs created before T minus the PR itself; acceptance = share of those merged before T.

```
fixed PR features: 2615 PRs
rq3b_all: 14923 rows, dropped for missing PR features 37 (0.0025) of 14923 non-vendored modified files; repos dropped rows {'blaizzy/mlx-vlm': 22, 'odigos-io/odigos': 15}
  bins {0: 2328, 1: 3646, 2: 4085, 3: 4827} share {0: 0.1564, 1: 0.2449, 2: 0.2744, 3: 0.3243}
rq3b_agent: 6300 rows, dropped for missing PR features 22 (0.0035) of 6300 non-vendored modified files; repos dropped rows {'blaizzy/mlx-vlm': 16, 'odigos-io/odigos': 6}
  bins {0: 949, 1: 1569, 2: 1683, 3: 2077} share {0: 0.1512, 1: 0.2499, 2: 0.2681, 3: 0.3308}
```

#### RQ3b full fix, agent files

| Model | Macro F1 | AUC-ROC |
|---|---|---|
| XGBoost | 0.3159 | 0.6049 |
| CatBoost | 0.3133 | 0.6063 |
| RandomForest | 0.3124 | 0.6010 |
| LogisticRegression | 0.2807 | 0.5813 |
| MLP | 0.2795 | 0.5811 |
| SVM | 0.2790 | 0.5771 |
| NaiveBayes | 0.2761 | 0.5618 |
| KNN | 0.2707 | 0.5349 |

Scott-Knott ESD (macro_f1): XGBoost (rank 1), CatBoost (rank 1), RandomForest (rank 1), LogisticRegression (rank 2), MLP (rank 2), SVM (rank 2), NaiveBayes (rank 2), KNN (rank 2)

Scott-Knott ESD (auc_roc): CatBoost (rank 1), XGBoost (rank 1), RandomForest (rank 1), LogisticRegression (rank 2), MLP (rank 2), SVM (rank 2), NaiveBayes (rank 3), KNN (rank 4)

Out-of-sample explanation: {"best_model": "XGBoost", "shap_model": "XGBoost", "selected_features": ["project_commit_velocity_3m", "file_modification_frequency_3m", "file_age_days", "contributor_prior_prs", "contributor_acceptance_rate", "project_open_prs_at_birth", "birth_day_of_week", "birth_hour"], "n": 6278, "repos": 160, "oos": {"macro_f1": 0.3204267784953063, "weighted_f1": 0.34611139866961116, "mcc": 0.11869297607201548, "precision_class0": 0.27715355805243447, "recall_class0": 0.1559536354056902, "f1_class0": 0.19959541469993258, "precision_class1": 0.39425287356321836, "recall_class1": 0.21861057998725303, "f1_class1": 0.2812628126281263, "precision_class2": 0.3216601815823606, "recall_class2": 0.2947118241235888, "f1_class2": 0.3075968992248062, "precision_class3": 0.40036014405762305, "recall_class3": 0.6422725084256139, "f1_class3": 0.49325198742836013, "auc_roc": 0.6050297527109703}, "oos_report": {"0": {"precision": 0.27715355805243447, "recall": 0.1559536354056902, "f1-score": 0.19959541469993258, "support": 949.0}, "1": {"precision": 0.39425287356321836, "recall": 0.21861057998725303, "f1-score": 0.2812628126281263, "support": 1569.0}, "2": {"precision": 0.3216601815823606, "recall": 0.2947118241235888, "f1-score": 0.3075968992248062, "support": 1683.0}, "3": {"precision": 0.40036014405762305, "recall": 0.6422725084256139, "f1-score": 0.49325198742836013, "support": 2077.0}, "accuracy": 0.36970372730168843, "macro avg": {"precision": 0.3483566893139091, "recall": 0.3278871369855365, "f1-score": 0.3204267784953063, "support": 6278.0}, "weighted avg": {"precision": 0.3591117537469331, "recall": 0.36970372730168843, "f1-score": 0.34611139866961116, "support": 6278.0}}, "shap_overall": {"file_modification_frequency_3m": 0.3464, "project_open_prs_at_birth": 0.201, "file_age_days": 0.1729, "project_commit_velocity_3m": 0.1619, "contributor_prior_prs": 0.1534, "contributor_acceptance_rate": 0.1388, "birth_day_of_week": 0.0911, "birth_hour": 0.0818}, "lime_overall": {"file_modification_frequency_3m": 0.1165, "project_open_prs_at_birth": 0.0532, "contributor_prior_prs": 0.0411, "file_age_days": 0.0366, "contributor_acceptance_rate": 0.0318, "birth_day_of_week": 0.0278, "project_commit_velocity_3m": 0.0272, "birth_hour": 0.0229}}

LIME runtime: {"calls": 5, "explanations": 200, "seconds_total": 1.540480375289917, "seconds_per_explanation": 0.007702401876449585}

#### RQ3b full fix, all files

| Model | Macro F1 | AUC-ROC |
|---|---|---|
| CatBoost | 0.3226 | 0.6153 |
| RandomForest | 0.3117 | 0.5983 |
| SVM | 0.3014 | 0.5874 |
| LogisticRegression | 0.3011 | 0.5850 |
| XGBoost | 0.3009 | 0.6011 |
| MLP | 0.2921 | 0.5871 |
| NaiveBayes | 0.2772 | 0.5791 |
| KNN | 0.2663 | 0.5367 |

Scott-Knott ESD (macro_f1): CatBoost (rank 1), RandomForest (rank 2), SVM (rank 3), LogisticRegression (rank 3), XGBoost (rank 3), MLP (rank 4), NaiveBayes (rank 5), KNN (rank 6)

Scott-Knott ESD (auc_roc): CatBoost (rank 1), XGBoost (rank 2), RandomForest (rank 2), SVM (rank 3), MLP (rank 3), LogisticRegression (rank 3), NaiveBayes (rank 3), KNN (rank 4)

Out-of-sample explanation: PENDING

LIME runtime: PENDING

