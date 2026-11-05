# Stage 7 Report — Office Frozen-Policy Case Studies

Stage 7 has been redesigned around a single frozen Office Recurrent PPO policy.
The implementation retains the five information conditions and adds the paired
branches required for the case studies: missing crowd information in a
non-CCTV region, counterfactual CCTV congestion corruption,
intention-induced freshness error under partial coverage, and a
normal-DRL comparison against stuck/oscillating behavior from the same frozen
policy. The Stage 6 reference planner remains an auxiliary comparator.

Stage 7 is **in progress**. The strengthened design passes 97 host tests, but
its 18-pedestrian Stage 6 reference and closed-loop Stage 7 episode have not
yet been generated. Earlier 12-pedestrian artifacts remain provenance only.
Hospital and
Warehouse are outside the currently authorized Stage 7 scope, and Stage 8 has
not started.

## 1. Experimental question

The following branches load the same Office checkpoint, use deterministic
inference, and keep every policy parameter frozen:

| Branch | Congestion input | Intention input | Primary comparison |
|---|---|---|---|
| Actual DRL | Stage 3 CCTV/FOV estimate | Stage 3 velocity prediction | — |
| Full-information DRL | exact current full-grid counts | Pegasus controller destination | Actual DRL |
| Congestion-error DRL | injected congestion error | full information | Full-information DRL |
| Intention-error DRL | full information | injected destination error | Full-information DRL |
| Combined-error DRL | both injected errors | both injected errors | Full-information DRL |

The case-study-specific paired branches are:

| Case | Correct/reference condition | Comparison condition |
|---|---|---|
| 1-1. Missing CCTV coverage | exact counterfactual coverage in the six-cell central non-CCTV region | Actual partial coverage |
| 1-2. CCTV sensing error | exact counts from the same two added direct-route CCTV cells | false-negative zero counts from occupied cells |
| 2. Intention/freshness error | scheduled correct destinations on a shared 240 s stale map | scheduled wrong destinations without anchor reset, then the same wrong destinations with reset |
| 3. Stuck/oscillation policy failure | successful Actual DRL | failing Full-information branch of the same frozen DRL |

This design isolates information-condition changes from learning. It does not
train, fine-tune, or duplicate policies. A better Full-information trajectory
is not assumed: the frozen policy was trained on the Actual observation
distribution, so privileged inputs can also create distribution shift.

The Stage 6 time-expanded reference planner is not a sixth DRL branch and is
not treated as globally optimal. Its shared navigation cost is used only to
help judge whether a DRL trajectory is inefficient.

Case 3 primarily compares two executions of the same frozen policy. The normal
Actual branch must reach the goal without a detected stuck or oscillation
window. The failing branch must exhibit both behaviors. A three-second window
is `stuck` when net displacement is at most 0.15 m and goal progress is at most
0.10 m. It is `oscillation` when net displacement is at most 0.25 m, accumulated
yaw travel is at least 0.75 rad, and the executed angular command reverses sign
at least twice outside a 0.15 rad/s deadband. Collision and failure remain
separate outcome evidence. The Stage 6 A* trajectory is retained only as an
auxiliary efficiency reference, not as the primary Case 3 comparator.

## 2. Information semantics

### Actual DRL

Actual DRL uses the validated Stage 3 costmap: exact CCTV cells, noisy virtual
robot-FOV updates in visible non-CCTV cells, constant-velocity intention
events, and timestamp-based 300-second linear freshness decay.

### Full-information DRL

At every Stage 3 tick, every non-CCTV cell receives the exact current
Ground-Truth pedestrian count as a direct replacement. CCTV cells retain their
existing exact update and cannot be overwritten. These direct observations
reset normal anchors, so the fully observed cells have freshness 1 at that
tick.

Full intention uses each Pegasus controller's
`current_destination_xyz_m`, maps it into the Stage 3 grid, and emits a new
event only when that pedestrian's destination changes. It does not use the
pedestrian's realized future trajectory.

### Injected errors

The controlled Office errors are:

- Congestion error: add 3 people to cell `[3, 3]` on top of its exact count.
- Intention error: replace the controller destinations for `office_04`,
  `office_10`, and `office_11` with direct-route/goal cell `[3,3]`. This is an
  explicitly labeled multi-person stress intervention used to make the
  intention/freshness effect observable with the frozen policy.
- Combined error: apply both changes to the same full-information state.
- CCTV case-study error: both paired branches add coverage over the same
  central region; the error branch reports false-negative zero counts in
  direct-route cells `[2,3]` and `[3,3]` from 0–20 s. A run is eligible for
  interpretation only if at least one corrupted sample has nonzero Ground
  Truth occupancy.
- Missing-coverage control: expose exact counts in the otherwise non-CCTV
  central corridor (`[2,2]`, `[2,3]`, `[3,2]`, `[3,3]`, `[4,2]`, `[4,3]`);
  the comparison branch retains the original sensing gap.

Normal and Actual branches continue to receive exact installed-CCTV counts at
every tick. The paired Case 1-2 sensor layout is identical: both branches use
the same counterfactual added-CCTV region, and only the error branch corrupts
the two direct-route counts. It is explicitly labeled
`counterfactual_added_cctv_corruption`; exact simulator counts remain in the
privileged log and are never mutated. This is a controlled robustness study,
not a claim that base VACaN CCTV observations are noisy.

Case 1-1 uses 18 deterministic pedestrians in every matched branch to increase
the probability of a decision-relevant hidden crowd without changing FOV or
camera range. It retains the original CCTV layout. `coverage_restored` is a privileged
counterfactual control for the central non-CCTV region, not a change to Actual
DRL. The exported metrics record the policy-visible estimate, exact crowd
count, visibility, last update source, whether the robot is inside each target
cell, and maximum crowd underestimation.

Case 2 does not use full congestion. All three branches retain the same limited
FOV and begin with the same affected-cell anchors aged by 240 s under the
unchanged 300 s linear decay. At 4 s, scheduled correct events target `[4,3]`
for `office_04`/`office_10` and `[1,2]` for `office_11`. The direct-only
ablation moves all three contributions to nonvisible central-left cell `[3,2]`
without resetting its
anchor. The full error branch applies the same wrong destinations and resets
the wrong cell's anchor timestamp and freshness.
Thus `direct-only - correct` measures the intention contribution and
`full-error - direct-only` measures the freshness-mediated addition. The
freshness treatment must reach an L2 difference of at least 0.25 before a
causal failure interpretation is eligible.
Ground-Truth validity is computed offline as
`1-min(abs(N_gt(t)-N_gt(anchor_t))/crowd_clip,1)` and is never given to the
policy.

The case figures contain three evidence panels: paired world-frame paths with
the maximum-separation segment marked, paired separation over simulation time,
and the relevant aggregate input/freshness signal. The 0.75 m candidate
threshold remains visible in the separation panel. The plotting code does not
rescale trajectory coordinates independently or fabricate route separation.

Congestion and Intention remain independently logged. Freshness is not an
independent intervention factor. A no-op branch is retained as a validation
control and must exactly reproduce Actual DRL.

## 3. Frozen policy and Ground-Truth separation

The selected policy is versioned at:

`checkpoints/stage5/office/recurrent_ppo.zip`

Its SHA-256 digest is
`d53598a14f636f2303d01da727f5204dc053135be1c5d3b1c46fc39da5408b30`.
The source is the selected 112,384-step Office Phase 4 social=1 checkpoint.
Policy parameters are fingerprinted before and after each comparison.

Privileged full information is supplied only to the explicitly named
Full-information and injected-error counterfactual branches. Actual DRL never
receives simulator Ground Truth. Policy trajectory records and privileged
pedestrian records remain separate.

## 4. Implementation

The optional `InterventionManager` remains inside the Stage 3 event pipeline.
Without a manager, Stage 3 and Stage 5 behavior is unchanged. The preserved
ordering is:

1. exact CCTV update, or explicitly isolated counterfactual corruption before
   the policy-visible CCTV mean;
2. non-CCTV linear decay;
3. noisy virtual-FOV observation;
4. congestion information condition and direct replacement;
5. intention information condition and event modification;
6. anchors, freshness, provenance, and audit export.

The runner resets every branch with the same scenario seed and resolved
pedestrian routes. Nova Carter remains physically controlled through the
differential-drive interface; counterfactual trajectories are not teleported
or forced to match after branching. Every closed-loop result is evaluated by
the same Stage 6 `SharedNavigationCost` implementation. Its primary subtotal
is trajectory efficiency (`time + 5*length + 0.05*smoothness`); social cost is
secondary, while collision and terminal failure remain explicit safety
penalties. The Stage 6 reference trajectory is rescored with the current
evaluator instead of reusing a stale cost stored in an older artifact.

## 5. Host decision-replay evidence

The redesigned nine-experiment decision replay passed with the same robot
state and incoming recurrent state. Each row uses its configured paired
baseline.

| Branch | Costmap L2 | Fused feature L2 | Actor hidden L2 | Action L2 | Critic delta |
|---|---:|---:|---:|---:|---:|
| Full-information | 8.917 | 7.573 | 0.164 | 0.000 | +0.088 |
| Congestion error | 9.487 | 6.014 | 0.162 | 0.000 | -0.012 |
| Intention error | 4.014 | 6.731 | 0.165 | 0.000 | +0.054 |
| Combined error | 10.301 | 8.754 | 0.268 | 0.000 | +0.043 |
| Coverage restored `[3,3]` vs Actual gap | 3.162 | 1.768 | 0.051 | 0.000 | -0.005 |
| CCTV congestion error vs Actual | 12.360 | 7.665 | 0.156 | 0.000 | +0.046 |
| Correct intention partial vs Actual | 6.667 | 6.000 | 0.163 | 0.000 | +0.046 |
| Intention/freshness error vs correct partial | 3.333 | 6.283 | 0.124 | 0.000 | +0.049 |
| No-op | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The clipped action did not change at this one decision even though internal
features, recurrent state, and critic value changed. This is a valid negative
decision-level result, not evidence that trajectories will remain identical.

## 6. Created and modified files

Stage 7 implementation and experiment definition:

- `configs/stage7/office.yaml`
- `src/vacan/counterfactual/{config,intervention,policy,dataset,case_studies}.py`
- `src/vacan/mapping/dynamic_costmap.py`
- `src/vacan/rl/stage5_env.py`
- `scripts/analyze_stage7_office_decisions.py`
- `scripts/isaacsim/run_stage7_office.py`
- `scripts/run_stage7_office.py`
- `scripts/summarize_stage7_office.py`
- `tests/test_stage7_counterfactual.py`

Repository packaging added for reproducibility:

- `.gitattributes`
- `checkpoints/README.md`
- `checkpoints/stage5/office/{recurrent_ppo.zip,metadata.yaml}`

The curated checkpoint, minimum reproduction inputs, and retained Office
Stage 7 archive are versioned as ordinary Git files. Other generated
experiment outputs remain ignored.

## 7. Commands and current validation

Host validation performed on 2026-11-04:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  src/vacan/counterfactual/intervention.py \
  src/vacan/counterfactual/config.py \
  src/vacan/mapping/dynamic_costmap.py \
  scripts/isaacsim/run_stage7_office.py
# 97 passed
```

Generate the population-matched Stage 6 auxiliary reference first:

```bash
.venv/bin/python scripts/run_stage6_office.py \
  --seed 15101 --output artifacts/stage6/office-strengthened \
  --gpu 1 --headless
```

Then use the printed Stage 6 run directory for the strengthened Stage 7 run:

```bash
STAGE6_RUN="$(find artifacts/stage6/office-strengthened -maxdepth 1 \
  -type d -name 'matched-seed15101-*' | sort | tail -1)"
test -n "$STAGE6_RUN"
.venv/bin/python scripts/run_stage7_office.py \
  --stage6-artifact "$STAGE6_RUN" \
  --seed 15101 --gpu 1 --headless
```

After it completes, summarize the newly printed run directory rather than the
superseded 2026-11-03 correction experiment:

```bash
.venv/bin/python scripts/summarize_stage7_office.py \
  artifacts/stage7/office/run-seed15101-<timestamp>
```

## 8. Generated assets and artifacts

The host replay currently writes eight experiment records, compressed policy
representations, provenance logs, and decision figures to the requested output
directory. A physical run will additionally write per-branch policy and
privileged trajectories, intervention JSONL, episode costs, trajectory arrays,
spatial-temporal query indexes, and figures.

All existing Stage 7 artifacts use the superseded 12-pedestrian designs. They
must not be reported as evidence for the strengthened experiment until a new
18-pedestrian run has completed.

## 9. Known limitations and acceptance status

- Only one Office seed and start/goal pair are configured for the first
  strengthened run; broader predeclared held-out evaluation remains required.
- The controlled errors occur at one time and fixed cells; broader error
  magnitudes and timings are not yet evaluated.
- Full-information intention is the controller's declared current destination,
  not privileged realized future motion.
- Zero immediate action distance is affected by action clipping and does not
  erase the measured internal-state differences.
- The Stage 6 reference can be infeasible or less efficient and is therefore
  an auxiliary comparator, not an optimality oracle.
- Hospital and Warehouse Stage 7 comparisons remain pending their own approved
  scope and environment-specific frozen policies.

The code marks a case as a candidate only when the erroneous/comparison branch
has higher shared cost and its path or outcome diverges. Otherwise it preserves
the result as negative; path differences are never fabricated.

Stage 7 must not be marked complete until the redesigned physical run passes,
the no-op equality check passes, all configured branches are evaluated with the
shared cost, and the report is updated with actual closed-loop results.
