# Stage 5 Implementation Report — In Progress

Stage 5 is not complete. The Recurrent PPO host implementation and isolated
Isaac Sim launcher pass host tests, and the first physical Office smoke training
run passed. Checkpoint reload, resume training, and integration evaluation in
Office, Hospital, and Warehouse pass. Full Phase 1 training produced successful
static navigation but regressed late in training. Best-checkpoint selection and
Phases 2–4 still have to be executed.

## 1. Verified prerequisites

- Isaac Sim 5.1.0, Isaac Lab 2.3.2, Pegasus 5.1.0.
- Stable-Baselines3 2.3.2 and sb3-contrib 2.3.0 in the VACaN `.venv`.
- TensorBoard 2.17.1 for PPO scalar and loss logging.
- Official Stage 4 maps validate for all environments:
  - Office: static `120×120` at 0.2 m; crowd `6×6` at 4 m.
  - Hospital: static `80×160` at 0.2 m; crowd `4×8` at 4 m.
  - Warehouse: static `80×100` at 0.2 m; crowd `4×5` at 4 m.
- Stage 3/4 host verification passed before Stage 5 implementation.

Isaac Sim's bundled Python does not contain sb3-contrib. The launcher does not
modify Isaac Sim or Isaac Lab. It invokes Isaac Sim's `python.sh` with VACaN's
`.venv` selected through `PYTHONEXE` and exposes the existing Isaac Lab and
Pegasus source directories through `PYTHONPATH`.

Isaac Sim exposes Gymnasium 1.2.1 ahead of the selected interpreter, while the
installed SB3 2.3.2 uses Gymnasium 0.29.1 APIs. The runtime selectively pins
only VACaN's Gymnasium 0.29.1 before importing SB3. It deliberately does not put
the full `.venv` site-packages directory on `PYTHONPATH`, because that would
replace Isaac's CUDA Torch 2.7 with VACaN's host-only CPU Torch 2.2.

## 2. Implemented architecture

- `VacanFeatureExtractor`: shared CNN over the two-channel `2×61×61` map and a
  separate MLP over the eight-element robot vector.
- sb3-contrib `MultiInputLstmPolicy` with a 128-unit LSTM, separate recurrent
  actor and critic state, continuous two-dimensional action distribution, and
  standard clipped PPO/GAE losses.
- Explicit deterministic interface accepting and returning LSTM hidden/cell
  state plus episode-start masks.
- `VacanNavigationEnv`: Gymnasium boundary that joins the physical Stage 2 Nova
  Carter/Pegasus runtime, exact Stage 3 costmap, and Stage 4 observation/action/
  reward implementation.
- 60 Hz physics is repeated twelve times per 5 Hz policy action. Costmap and
  policy timing use simulation steps, including the unchanged 300-second decay.
- Policy observations contain only `map` and `robot`. Freshness remains internal
  costmap state, and privileged pedestrian state is used only by costmap/reward
  evaluation and diagnostics.
- Four configurable curriculum phases. Each run resolves a deterministic
  pedestrian subset and can resume from the preceding phase checkpoint.
- Isolated smoke/train/evaluate launcher, periodic checkpoints, TensorBoard
  path, Monitor CSV, evaluation JSONL, hashes, seeds, versions, and source
  revision metadata.
- Isaac Sim and PPO output is streamed to the terminal in real time and written
  to the run-specific `isaac-sim-training.log` at the same time.

## 3. Files

- `configs/stage5/recurrent_ppo.yaml`
- `src/vacan/rl/stage5.py`
- `src/vacan/rl/stage5_config.py`
- `src/vacan/rl/stage5_env.py`
- `scripts/run_stage5.py`
- `scripts/isaacsim/run_stage5_training.py`
- `tests/test_stage5_recurrent_ppo.py`
- `scripts/isaacsim/run_stage2_episode.py` (only `parse_known_args` compatibility
  change so the validated runtime can accept Stage 5 caller-owned arguments)

## 4. Tests and actual results

```bash
.venv/bin/pytest -q tests/test_stage5_recurrent_ppo.py
# 4 passed in 6.60s

.venv/bin/pytest -q
# 58 passed in 4.44s

.venv/bin/python scripts/run_stage5.py \
  --mode smoke --scenario office --phase dynamic_crowd --headless --dry-run
# PASS: emitted the isolated Isaac Sim command and VACaN .venv selection

.venv/bin/python scripts/run_stage5.py \
  --mode smoke --scenario office --phase dynamic_crowd \
  --seed 5101 --gpu 1 --headless
# PASS: 512 Recurrent PPO steps and policy-latest.zip saved
```

The tests cover the exact observation keys, rejection of Freshness/Ground Truth
inputs, CNN/MLP fusion, recurrent forward pass, hidden-state reset, a finite PPO
gradient update, checkpoint save/load, and deterministic reloaded inference.

## 5. Physical smoke result

- Artifact: `artifacts/stage5/smoke-office-dynamic_crowd-seed5101-20261101T052039Z`
- Status: pass
- PPO steps: 512 (two 256-step rollouts)
- PPO elapsed time: 129.916 s
- Reported policy throughput: 3 steps/s
- Mean episode length: 61 policy steps
- Mean episode return: -25.2
- Approximate KL: 0.00040172925
- Policy-gradient loss: -0.00124
- Value loss: 12.7
- Entropy loss: -2.84
- Explained variance: 0.0187
- Checkpoint: `policy-latest.zip` (5,924,758 bytes)
- TensorBoard event and Monitor CSV: created

The policy archive reloads as `RecurrentPPO`, reports 512 timesteps and a
128-unit actor LSTM. These smoke statistics establish a functioning optimizer
and checkpoint path; they do not establish learned navigation.

The next checkpoint reload command is:

```bash
.venv/bin/python scripts/run_stage5.py \
  --mode evaluate --scenario office --phase dynamic_crowd \
  --seed 15101 --gpu 1 --headless \
  --checkpoint artifacts/stage5/smoke-office-dynamic_crowd-seed5101-20261101T052039Z/policy-latest.zip
```

This deterministic checkpoint reload evaluation passed and produced:

- Artifact: `artifacts/stage5/evaluate-office-dynamic_crowd-seed15101-20261101T052547Z`
- Held-out seed: 15101
- Deterministic steps: 600 (7,200 physics steps; 120 seconds simulation time)
- Episode return: -20.1765
- Final goal distance: 5.0496 m
- Collision: false
- Goal reached: false
- Outcome: timeout
- Wall-clock evaluation time: 153.358 s

Checkpoint reload and deterministic recurrent inference therefore pass. The
smoke checkpoint did not learn navigation and must not be reported as a
successful navigation policy.

A physical resume-training validation also passed:

- Artifact: `artifacts/stage5/train-office-dynamic_crowd-seed5101-20261101T053105Z`
- Source checkpoint steps: 512
- Additional requested steps: 256
- Saved checkpoint steps: 768
- PPO update count: 12
- Elapsed training time: 68.538 s
- Reported throughput: 3 policy steps/s
- Mean episode length: 58.5 policy steps
- Mean episode return: -25.0

The resumed checkpoint is saved in the new run directory and reloads with a
128-unit LSTM. During this validation, SB3 restored the source checkpoint's
TensorBoard directory and appended the resume metrics there. The runtime now
overrides that serialized path so subsequent resumed runs keep TensorBoard
artifacts in their own output directory.

The resumed 768-step checkpoint was then evaluated in the remaining official
environments with held-out seed 15101:

| Environment | Runtime status | Steps | Return | Final goal distance | Collision | Outcome |
|---|---|---:|---:|---:|---|---|
| Office | pass | 600 | -20.1765 | 5.0496 m | false | timeout |
| Hospital | pass | 600 | -49.0245 | 3.5943 m | false | timeout |
| Warehouse | pass | 600 | -35.4159 | 5.2125 m | false | timeout |

Hospital took 148.700 wall-clock seconds and Warehouse 144.872 seconds. Each
episode covered 7,200 physics steps and 120 seconds of simulation time. The
Isaac Sim/Pegasus/costmap/policy integration therefore executes in all three
environments, but navigation success rate is 0/3 for this deliberately tiny
checkpoint.

### Phase 1 pilot

A fresh-policy 4,096-step `static_navigation` pilot passed:

- Artifact: `artifacts/stage5/train-office-static_navigation-seed5101-20261101T054851Z`
- Pedestrians: 0; the two-channel observation structure was preserved.
- Training elapsed time: 792.291 s
- Throughput: 5.17 policy steps/s (SB3 displayed 5 steps/s)
- Completed episodes: 17
- Mean episode return: -20.0934
- Mean episode length: 229.235 policy steps
- Return range: -42.8123 to 20.1224
- Episodes reaching the 600-step limit: 2
- Final logged mean return: -20.1
- Final logged mean episode length: 229
- Final approximate KL: 0.0033731
- Final policy-gradient loss: -0.00747
- Final value loss: 15.7
- Final explained variance: 0.133

One episode had positive return 20.1224 and is consistent with a possible goal
completion, but the pilot's Monitor CSV did not include terminal reason fields.
It is therefore not counted as a verified success. The runtime now records
`goal_reached`, `collision`, `timeout`, `termination_reason`, and final goal
distance in all subsequent training Monitor rows.

The pilot does not show a clear return improvement: the logged mean return was
-20.1 at both 256 and 4,096 steps, with substantial episode variability. A
deterministic checkpoint evaluation is required before committing to the full
100,000-step Phase 1 run.

The held-out deterministic evaluation subsequently passed:

- Artifact: `artifacts/stage5/evaluate-office-static_navigation-seed15101-20261101T061019Z`
- Checkpoint steps: 4,096
- Seed: 15101
- Goal reached: true
- Collision: false
- Timeout: false
- Termination reason: goal
- Policy steps: 296
- Physics steps: 3,551
- Approximate simulated completion time: 59.18 s
- Final goal distance: 0.49955 m (inside the 0.5 m tolerance)
- Episode return: 30.8441
- Wall-clock evaluation time: 57.182 s

This is evidence of successful navigation for one held-out Office seed. It is
not yet a multi-seed success-rate result and does not establish robustness or
curriculum convergence.

### Full Phase 1 training

The 4,096-step pilot checkpoint was resumed for 96,000 additional steps. The
run passed and saved a 100,096-step checkpoint:

- Artifact: `artifacts/stage5/train-office-static_navigation-seed5101-20261101T062400Z`
- Training steps: 100,096 total
- Additional run time: 20,261.429 s (5 h 37 min 41 s)
- Additional-run throughput: 4.74 policy steps/s
- Completed episodes in the resumed run: 848
- Goals: 627 (73.94%)
- Collisions: 217 (25.59%)
- Timeouts: 4 (0.47%)
- Mean episode return: 18.0640
- Mean episode length: 113.16 policy steps
- Last 100 episodes: 77% goal, 22% collision, 1% timeout

Training was not monotonic. Episode quartiles were:

| Episode range | Mean return | Success | Collision |
|---|---:|---:|---:|
| 1–212 | -4.969 | 33.5% | 66.0% |
| 213–424 | 21.295 | 81.1% | 17.9% |
| 425–636 | 30.387 | 95.8% | 4.2% |
| 637–848 | 25.543 | 85.4% | 14.2% |

TensorBoard's highest rolling mean return was 32.7194 at step 88,320; the final
rolling mean return was 21.2689. At the final update, approximate KL was 0.06825,
clip fraction 0.218, value loss 22.2, and explained variance -0.118. The late
drop in success and negative final explained variance indicate policy/critic
regression. Therefore `policy-latest.zip` must not automatically be selected as
the Phase 1 best checkpoint.

Periodic checkpoints exist at 14,096 through 94,096 steps in 10,000-step
increments. The 84,096-step, 94,096-step, and final 100,096-step policies should
be compared using identical deterministic held-out seeds before Phase 2.

The three deterministic comparisons on Office seed 15101 produced:

| Checkpoint | Goal | Collision | Travel time | Policy steps | Final distance | Return |
|---:|---|---|---:|---:|---:|---:|
| 84,096 | true | false | 9.4 s | 47 | 0.49797 m | 32.96961 |
| 94,096 | true | false | 14.4 s | 72 | 0.49131 m | 32.67180 |
| 100,096 | true | false | **9.0 s** | **45** | 0.49190 m | 32.96745 |

All three pass the held-out deterministic navigation check. Following the
predeclared comparison order (success, collision, then travel time), the
100,096-step policy is the provisional Phase 1 checkpoint. The 84,096-step
policy is a close alternative and has a marginally higher return. This selection
is only for one fixed start/goal and seed; it is not a robust multi-start success
rate comparison.

### Phase 2 pilot

The provisional Phase 1 checkpoint was resumed for a 4,096-step
`low_density` pilot:

- Artifact: `artifacts/stage5/train-office-low_density-seed5101-20261101T132901Z`
- Cumulative checkpoint steps: 104,192
- Low-density pedestrians: 2 (35% of the six-person Office configuration,
  rounded to the nearest person count)
- Elapsed training time: 978.235 s
- Completed episodes: 82
- Goals: 81 (98.78%)
- Collisions: 1 (1.22%)
- Timeouts: 0
- Mean episode return: 31.8715
- Mean episode length: 49.70 policy steps (9.94 simulated seconds)
- Final rolling mean return: 30.1
- Final rolling mean episode length: 54.2
- Final approximate KL: 0.02939
- Final clip fraction: 0.173
- Final explained variance: 0.44
- Final value loss: 56.8

The training-run results show that the Phase 1 behavior was retained while two
moving pedestrians and the Stage 3 dynamic costmap were active. The final value
loss is noisy, so a held-out deterministic evaluation is required before longer
Phase 2 training. These statistics are training episodes, not an independent
success-rate estimate.

The held-out deterministic Phase 2 evaluation passed:

- Artifact: `artifacts/stage5/evaluate-office-low_density-seed15101-20261102T061625Z`
- Checkpoint steps: 104,192
- Goal reached: true
- Collision: false
- Timeout: false
- Policy steps: 46
- Travel time: 9.2 simulated seconds
- Final goal distance: 0.49103 m
- Episode return: 32.93793
- Minimum pedestrian distance: 0.76031 m
- Personal-space penalty steps: 2 (threshold 0.8 m)

Phase 2 pilot acceptance therefore passes for this held-out episode. It retains
fast goal navigation with two moving pedestrians, but the brief personal-space
violation remains relevant for later social-cost evaluation. This is one fixed
start/goal and pedestrian-route configuration, not a robust success-rate
estimate.

### Phase 3 pilot

The Phase 2 checkpoint was resumed for a 4,096-step `dynamic_crowd` pilot with
all six Office pedestrians:

- Artifact: `artifacts/stage5/train-office-dynamic_crowd-seed5101-20261102T062326Z`
- Cumulative checkpoint steps: 108,288
- Pedestrians: 6
- Elapsed training time: 1,108.226 s
- Completed episodes: 76
- Goals: 75 (98.68%)
- Collisions: 1 (1.32%)
- Timeouts: 0
- Mean episode return: 28.6769
- Mean episode length: 53.38 policy steps
- Final rolling mean return: 29.1
- Final rolling episode length: 52.9
- Final approximate KL: 0.03467
- Final clip fraction: 0.163
- Final explained variance: 0.706
- Final value loss: 0.416

The training-run results retain high success with full pedestrian density, but
they are not an independent evaluation. A held-out deterministic Phase 3
episode is required before proceeding to the final partial-coverage/noise phase.

The held-out deterministic Phase 3 evaluation produced:

- Artifact: `artifacts/stage5/evaluate-office-dynamic_crowd-seed15101-20261102T064801Z`
- Checkpoint steps: 108,288
- Goal reached: true
- Collision: false
- Timeout: false
- Policy steps: 47
- Travel time: 9.4 simulated seconds
- Final goal distance: 0.49579 m
- Episode return: 29.27961
- Minimum pedestrian distance: 0.37599 m
- Personal-space penalty steps: 10 (threshold 0.8 m)

The Phase 3 checkpoint passes goal/collision/timeout navigation acceptance for
this held-out episode. It does not pass a strong social-safety interpretation:
the robot passed substantially inside the configured personal-space radius for
2.0 simulated seconds. This must remain separately reported and investigated
before claiming robust dynamic pedestrian avoidance.

### Phase 4 definition

The initial curriculum made Phase 3 and Phase 4 observation settings identical.
Per user direction, Phase 4 keeps the 90-degree FOV and 6 m observation range,
doubles the Office population, and increases measurement noise and the direct
observation overlap threshold. The Stage 3 update algorithm, partial CCTV
regions, exact CCTV counts, intention logic, and 300-second linear decay remain
unchanged:

| Parameter | Phase 3 | Phase 4 |
|---|---:|---:|
| Pedestrians | 6 | 12 |
| Horizontal FOV | 90 degrees | 90 degrees |
| Maximum range | 6.0 m | 6.0 m |
| Position noise std. | 0.15 m | 0.30 m |
| Velocity noise std. | 0.10 m/s | 0.20 m/s |
| Minimum cell overlap | 0.20 | 0.40 |

The six additional pedestrians receive unique IDs and start at least 0.8 m from
all other initial pedestrian positions. Their initial and cyclic routes consist
only of points from the existing Stage 1 NavMesh-validated route cache; no
straight-line or unvalidated route fallback is introduced. Each Phase 4 run
saves the expanded population in `resolved-stage1.yaml` and
`resolved-routes.json`.

Every run writes the effective parameters to `resolved-stage3.yaml` and records
the overrides in `summary.json`. Evaluation logging now also includes complete
robot state, executed path length, per-episode minimum pedestrian distance and
social-violation steps, policy observations, recurrent hidden/cell states, and
mean/p95 inference latency. Ground Truth pedestrians remain in the privileged
`info` record and never enter the policy observation.

The 4,096-step Phase 4 pilot executed with the intended settings:

- Artifact: `artifacts/stage5/train-office-partial_coverage_noise-seed5101-20261102T071023Z`
- Cumulative checkpoint steps: 112,384
- Pedestrians: 12
- FOV/range: 90 degrees / 6.0 m
- Position/velocity noise: 0.30 m / 0.20 m/s
- Minimum overlap ratio: 0.40
- Elapsed training time: 1,263.875 s
- Completed episodes: 76
- Goals: 74 (97.37%)
- Collisions: 2 (2.63%)
- Timeouts: 0
- Mean episode return: 28.0222
- Mean episode length: 53.58 policy steps
- Mean executed path length: 5.759 m
- Final rolling mean return: 28.5
- Final approximate KL: 0.04136
- Final clip fraction: 0.144
- Final explained variance: 0.224
- Final value loss: 2.08

Navigation success remained high, but social safety did not pass. Every one of
the 76 training episodes entered the 0.8 m personal-space radius. The mean was
11.32 violation steps per episode, mean per-episode minimum distance was 0.3782
m, and the global minimum was 0.0477 m. Animated Pegasus people may not produce
the same rigid-contact signal as static geometry, so a false collision flag is
not sufficient evidence of safe pedestrian clearance. Ground Truth distance is
the authoritative social-safety diagnostic here.

The held-out deterministic Phase 4 evaluation produced:

- Artifact: `artifacts/stage5/evaluate-office-partial_coverage_noise-seed15101-20261102T073951Z`
- Goal reached: true
- Collision: false
- Timeout: false
- Travel time: 10.0 simulated seconds
- Policy steps: 50
- Executed path length: 5.5435 m
- Final goal distance: 0.49912 m
- Episode return: 31.36098
- Minimum pedestrian distance: 0.41134 m
- Personal-space violation steps: 9
- CUDA cold-start inference latency: 412.13 ms
- Steady-state mean inference latency: 2.67 ms
- Steady-state p95 inference latency: approximately 2.80 ms
- Evaluation JSONL size: 4.6 MB

The recurrent episode-start convention is correct: the first record is true and
all subsequent records are false. Navigation acceptance passes, while social
safety again fails. The original all-sample latency mean of 10.86 ms is retained
for audit but is dominated by the first CUDA initialization; subsequent runs
now report cold-start and steady-state latency separately.

Because both Phase 3 and Phase 4 held-out evaluations violated the 0.8 m
personal-space radius, a controlled Phase 4 reward experiment was authorized.
Only the social coefficient changes from 1.0 to 3.0; goal, progress, collision,
time, smoothness, observations, crowd population, PPO settings, and checkpoint
initialization remain unchanged. Each run records this override in
`summary.json` and the complete effective reward in `resolved-reward.json`.

The 4,096-step social-coefficient experiment and held-out evaluation completed:

| Metric | social=1 baseline | social=3 experiment |
|---|---:|---:|
| Training success | 97.37% | 90.00% |
| Training collision | 2.63% | 10.00% |
| Training mean minimum distance | 0.378 m | 0.401 m |
| Training mean violation steps | 11.32 | 12.97 |
| Held-out goal | true | true |
| Held-out travel time | 10.0 s | 9.0 s |
| Held-out path length | 5.543 m | 5.004 m |
| Held-out minimum distance | 0.411 m | 0.277 m |
| Held-out violation steps | 9 | 8 |

The higher social coefficient did not produce a consistent safety improvement:
it reduced held-out violation duration by one step but moved substantially closer
to a pedestrian, increased training collision rate, and reduced training success.
The social=3 checkpoint is rejected. The selected Phase 4 candidate remains the
112,384-step social=1 checkpoint at
`train-office-partial_coverage_noise-seed5101-20261102T071023Z/policy-latest.zip`.
The active configuration was restored to social=1.0; the experiment retains its
resolved reward and config hash in its own artifact directories.

The selected social=1 checkpoint was evaluated with 12 pedestrians in all three
environments:

| Scenario | Goal | Collision | Outcome | Time | Path length | Final distance | Min person distance | Social steps |
|---|---|---|---|---:|---:|---:|---:|---:|
| Office | true | false | goal | 10.0 s | 5.543 m | 0.499 m | 0.411 m | 9 |
| Hospital | false | false | timeout | 120.0 s | 61.482 m | 4.891 m | 0.315 m | 22 |
| Warehouse | false | true | collision | 105.8 s | 50.587 m | 5.191 m | 0.055 m | 68 |

Steady-state mean inference latency was 2.52 ms in Hospital and 2.54 ms in
Warehouse, with p95 below 2.88 ms. Runtime integration and detailed logging pass,
but navigation success is 1/3. The policy was trained only in Office and does
not generalize to the Hospital or Warehouse start/goal and static geometry.
Stage 5 three-environment acceptance therefore fails; these results must not be
reported as a completed navigation policy.

### Hospital Phase 4 fine-tuning and held-out result

The selected Office Phase 4 checkpoint was resumed for 4,096 additional steps
in Hospital with the unchanged 90-degree FOV, 6 m observation range, and 12
pedestrians. The resulting checkpoint has 116,480 cumulative steps. Simulator
execution and checkpoint serialization passed, but the Hospital training
episodes did not demonstrate task success:

| Training episodes | Goals | Collisions | Timeouts | Success rate | Collision rate |
|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 12 | 4 | 0% | 75% |

The deterministic held-out seed 15101 evaluation also failed the navigation
criterion:

| Goal | Collision | Outcome | Time | Path length | Final distance | Min person distance | Social steps |
|---|---|---|---:|---:|---:|---:|---:|
| false | true | collision | 10.6 s | 5.407 m | 2.144 m | 0.402 m | 14 |

The evaluation process returned `status: pass`, which means the evaluation
software completed without an exception; it does **not** mean that navigation
succeeded. The terminal action was `[0.55, 0.9]`, and the robot collided after
53 control steps. The high positive rolling reward printed near the end of the
resumed SB3 log includes the prior Office episode buffer and is therefore not a
Hospital success metric; the Hospital `monitor.csv` contains zero goal episodes.

This direct full-difficulty transfer is rejected. Warehouse fine-tuning should
not start from this failed Hospital checkpoint. Hospital needs an
environment-specific curriculum beginning with static navigation, followed by
low-density, dynamic-crowd, and Phase 4 settings, with deterministic held-out
acceptance at each boundary. A later mixed-environment rehearsal/evaluation is
required to check catastrophic forgetting of Office.

## 6. Known limitations and acceptance status

Stage 5 now uses one independent Recurrent PPO policy per environment. New runs
are stored below `artifacts/stage5/policies/<scenario>/`, write a
`policy-metadata.json` ownership record, and reject resume/evaluation when the
checkpoint environment differs from `--scenario`. Existing checkpoints remain
usable only when their adjacent `summary.json` proves matching environment
ownership. Consequently, Hospital and Warehouse curricula start from fresh
models and no longer inherit the Office policy; checkpoints are transferred
only between curriculum phases of the same environment.

Fresh environment-specific Phase 1 pilots are launched independently:

```bash
.venv/bin/python scripts/run_stage5.py --mode train --scenario office \
  --phase static_navigation --seed 5101 --total-steps 4096 --gpu 1 --headless
.venv/bin/python scripts/run_stage5.py --mode train --scenario hospital \
  --phase static_navigation --seed 5101 --total-steps 4096 --gpu 1 --headless
.venv/bin/python scripts/run_stage5.py --mode train --scenario warehouse \
  --phase static_navigation --seed 5101 --total-steps 4096 --gpu 1 --headless
```

The Hospital and Warehouse commands intentionally omit `--checkpoint`; this
prevents either policy from inheriting Office parameters. Later phases use only
the immediately preceding checkpoint from the same scenario directory.

The independent 4,096-step static-navigation pilots completed on 2026-11-02:

| Scenario | Episodes | Goals | Collisions | Timeouts | Success rate | Mean return | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hospital | 46 | 0 | 46 | 0 | 0.0% | -22.061 | 88.7 |
| Warehouse | 7 | 1 | 0 | 6 | 14.3% | -7.520 | 571.7 |

Both runs passed software execution, saved environment-owned checkpoints, and
used zero pedestrians as required by Phase 1. They do not yet pass navigation
acceptance: Hospital always collided, while Warehouse usually timed out. Phase
2 must not start. Because the pilot training evidence already establishes that
both policies are under-trained, repeating held-out evaluation at this point
would not repair or meaningfully select either policy and would unnecessarily
reuse the validation seed. Each environment instead resumes its own Phase 1
checkpoint for the remaining 95,904 steps. Deterministic seed 15101 evaluation
is performed only after the 100,000-step Phase 1 checkpoint is available.

The two environment-specific Phase 1 runs were resumed for 95,904 steps and
finished at 100,096 cumulative steps:

| Scenario | Extension episodes | Goals | Collisions | Timeouts | Overall success | Last-100 success | Last-100 collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hospital | 725 | 58 | 630 | 37 | 8.0% | 43.0% | 55.0% |
| Warehouse | 1,423 | 1,339 | 61 | 23 | 94.1% | 100.0% | 0.0% |

Hospital improved materially over training: success increased from 0% in the
first 100 extension episodes to 43% in the last 100 and 56% in the last 50.
However, the last 100 episodes still contain 55 collisions, so the policy is
not yet a stable Phase 1 solution. Warehouse converged strongly: all last 100
episodes reached the goal without collision or timeout. The 100,096-step latest
checkpoints are retained as deterministic evaluation candidates, but only
Warehouse currently has strong training evidence for Phase 1 acceptance.

- The physical smoke run passed, but GPU peak memory was not yet recorded.
- The 512-step smoke policy has not demonstrated meaningful navigation. Its
  held-out deterministic Office episode timed out.
- Phase 1 checkpoint comparison and Phase 2–4 pilots pass navigation checks for
  the fixed Office start/goal. Phase 3/4 social safety and robust multi-start
  evaluation remain unresolved.
- The final Phase 4 cross-environment evaluation completed: Office succeeded,
  Hospital timed out, and Warehouse collided. A subsequent direct Hospital
  Phase 4 fine-tune also produced 0/16 training successes and a held-out
  collision.
- The detailed logger split is implemented, while the initial runtime evaluation
  currently writes compact action/reward/info JSONL; complete policy distribution
  and value logging remains pending.
- Physical checkpoint reload, resume training, three-environment integration,
  and one held-out static-navigation success pass. Full curriculum and robust
  multi-seed evaluation have not passed.

Stage 5 remains **In Progress** until the remaining acceptance runs pass. Stage
6 has not been started.
