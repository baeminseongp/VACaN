# Stage 6 Implementation Report — Office Integration Verified

Stage 6 is **not complete overall**, but the authorized Office scope now passes
its matched physical integration and reproducibility checks. The host-side
full-information reference planning, shared navigation cost, comparison,
failure dataset, and debug visualization pipeline is implemented and verified.
Hospital and Warehouse remain pending until their environment-specific Stage 5
policies are frozen.

## 1. Implemented components

- Heading-aware Time-Expanded A* over `(time,row,column,heading)` states.
- Static collision avoidance using the Stage 4 inflated/unknown obstacle mask.
- Dynamic pedestrian occupancy from current privileged Pegasus position and
  velocity with constant-velocity extrapolation.
- No actual future simulator trajectory or oracle future state is used.
- Bounded waypoint controller that executes through the existing Nova Carter
  differential-drive interface; no teleportation path exists.
- Matched Office runner that resets DRL and Reference executions with the same
  seed and verifies initial robot/pedestrian position equality.
- Frozen deterministic Office Recurrent PPO loading with explicit recurrent
  hidden-state handling and unchanged Stage 3 observed costmap rules.
- One shared cost evaluator for actual DRL and Reference trajectories.
- Timestamp-interpolated trajectory divergence analysis; arrays are not aligned
  by index.
- JSON/JSONL/NPZ failure dataset plus top-down trajectory and cost plots.
- Privileged pedestrian state is stored in separate Ground Truth JSONL files,
  not in the DRL policy observation record.
- Each physical DRL decision now records the deterministic Gaussian action
  mean/standard deviation, selected-action log probability, critic value
  estimate, episode-start flag, and separate actor/critic LSTM input/output
  states. The logged observation is the exact pre-step observation that
  produced that action. `policy_requested_action` is kept separate from
  `action`, which is the acceleration-limited command physically applied by
  Stage 2. These fields use only policy-visible history.

## 2. Reference assumptions

The primary reference uses current Ground Truth pedestrian positions and
velocities at each replan. Predicted occupancy is
`p(t + tau) = p(t) + v(t) * tau`. This is a reproducible constant-velocity
reference, not an actual-future oracle and not a claim of global optimality.

Planning uses a 0.8 s lattice step, 40 s horizon, eight heading bins, 0.8 m
pedestrian exclusion radius, and 1.0 s replanning interval. The chosen lattice
step bounds an eight-connected move to 0.354 m/s and a 45-degree rotation to
0.982 rad/s, within the Office limits of 0.6 m/s and 1.0 rad/s. The Stage 4 map
already includes the configured robot footprint and safety margin.

Pegasus pedestrians use deterministic resolved routes and limited priority
yielding. They do not implement SFM/ORCA and do not explicitly react to Nova
Carter, so matched resets reproduce their initial conditions and controller
configuration.

## 3. Shared navigation cost

The evaluator is separate from the PPO reward:

`J = 1*time + 2*length + 50*collision + 10*social + 0.1*smoothness + failure`

- Time is seconds of actual execution.
- Length is meters of actual robot displacement.
- Collision counts distinct collision-event rising edges.
- Social cost integrates normalized personal-space penetration below 0.8 m over
  simulation time.
- Smoothness sums normalized changes in executed linear/angular commands.
- An unsuccessful episode receives a 120-unit terminal failure penalty.
- `Delta_J = J_DRL - J_Reference`; inefficiency requires `Delta_J > 1e-6`.

The same `SharedNavigationCost` instance evaluates both methods. A positive
Delta J identifies relative inefficiency only; it does not assign Congestion,
Intention, or Freshness causality.

## 4. Files

- `configs/stage6/office.yaml`
- `src/vacan/reference/config.py`
- `src/vacan/reference/planner.py`
- `src/vacan/reference/cost.py`
- `src/vacan/reference/comparison.py`
- `src/vacan/reference/dataset.py`
- `src/vacan/reference/aggregate.py`
- `src/vacan/rl/stage5.py` (recurrent diagnostic inference interface)
- `scripts/run_stage6_office.py`
- `scripts/isaacsim/run_stage6_office.py`
- `scripts/summarize_stage6.py`
- `scripts/analyze_stage6_diagnostics.py`
- `tests/test_stage6_reference.py`
- `tests/test_stage5_recurrent_ppo.py`

## 5. Tests and actual results

```bash
.venv/bin/python -m pytest \
  tests/test_stage6_reference.py tests/test_stage5_recurrent_ppo.py -q
# 18 passed in 4.56s

.venv/bin/python -m pytest -q
# 72 passed in 4.72s

.venv/bin/python -m py_compile \
  scripts/run_stage6_office.py scripts/isaacsim/run_stage6_office.py \
  src/vacan/reference/*.py
# PASS

.venv/bin/python scripts/run_stage6_office.py \
  --seed 15101 --gpu 1 --headless --dry-run
# PASS: Office frozen checkpoint and isolated Isaac Sim command resolved
```

The host planner was also run against the actual Office Stage 4 map from
`(0,-2.5)` to `(0,2.5)` with no pedestrians. It returned a feasible 26-state
path after expanding 39 states. This is a host planning check, not evidence of
physical reference execution.

## 6. Physical Office results

The matched Office evaluation was executed twice with validation seed 15101.
The second run reproduced the first run exactly: DRL trajectory, Reference
trajectory, and comparison JSON SHA-256 hashes are identical. An independent
test-seed 25101 matched episode also passed.

| Metric | Frozen DRL | Full-information Reference |
|---|---:|---:|
| Goal reached | true | true |
| Collision events | 0 | 0 |
| Travel time | 9.967 s | 31.550 s |
| Executed path length | 5.540 m | 5.469 m |
| Social violation integral | 0.283 s | 0.925 s |
| Smoothness measure | 14.789 | 28.768 |
| Shared navigation cost | 25.357 | 54.611 |

`Delta_J = J_DRL - J_Reference = -29.254`; therefore this episode is classified
as **not inefficient relative to this Reference**. Both methods succeeded. This
does not establish policy optimality: the Reference replanned 32 times, returned
infeasible for 11 calls while pedestrians occupied constrained local regions,
and stopped until a later feasible replan. Its longer execution and larger
social/smoothness costs explain the negative Delta J.

Matched initialization errors were exactly zero for robot position, robot
velocity, pedestrian positions, and pedestrian velocities. The run used 12
Pegasus pedestrians. A privileged-field scan of `drl-trajectory.jsonl` passed,
and DRL Ground Truth remains in `ground-truth-drl.jsonl`.

Successful artifacts:

- `artifacts/stage6/office/matched-seed15101-20261103T152437Z`
- `artifacts/stage6/office/matched-seed15101-20261103T152656Z`
- `artifacts/stage6/office/matched-seed25101-20261103T152951Z`
- `artifacts/stage6/office/matched-seed15101-20261103T154116Z` (final aligned
  policy observation/action/distribution/value logging validation)

The final diagnostic-logging validation completed in 111.06 wall-clock seconds and
reproduced all 51 core DRL trajectory records (timestamp, position, yaw,
action, collision, and goal result) exactly. All 50 policy decisions contain
the exact pre-step observation, requested action, applied command, actor/critic
state, and new diagnostics. Independent replay of the first logged decision
reproduced its requested action, critic value, action mean, and action standard
deviation. The critic values ranged from 18.288 to 23.793 and the action
standard deviations ranged from 0.841 to 0.876. These are descriptive policy
outputs, not calibrated uncertainty or causal explanations.

Seed-deduplicated Office results over validation 15101 and test 25101 are:

| Aggregate metric | Result |
|---|---:|
| Unique matched episodes | 2 |
| DRL success / collision | 100% / 0% |
| Reference success / collision | 100% / 0% |
| Mean DRL time | 10.042 s |
| Mean DRL path length | 5.583 m |
| Mean DRL cost | 25.798 |
| Mean Reference time | 31.550 s |
| Mean Reference path length | 5.469 m |
| Mean Reference cost | 54.611 |
| Mean / median Delta J | -28.813 / -28.813 |
| Delta J population standard deviation | 0.441 |
| Inefficient episode fraction | 0% |
| Reference infeasible replan rate | 34.375% |

The aggregate is saved at `artifacts/stage6/office/aggregate-summary.json`.
The route controller uses fixed resolved Stage 1 routes, so changing the Stage
6 seed changes observed-costmap noise but does not currently change pedestrian
initial routes. The identical Reference metrics across these seeds are therefore
expected and must not be presented as independent pedestrian configurations.

Candidate information diagnostics are saved as `diagnostic-candidates.json` in
the selected validation/test run directories. Both runs produce one continuous
candidate interval spanning the episode, with crowd-estimation errors and
intention discrepancies. These broad intervals are traceable Stage 7 inputs,
not evidence that either information component caused a navigation failure.

The first attempted physical run inside the restricted tool sandbox failed
before environment construction because CUDA and the official NVIDIA asset URL
were unavailable. Its exception is retained under
`matched-seed15101-20261103T152321Z`. Re-running outside that sandbox with the
same repository command succeeded; no simulator fallback was used.

Reproduction command:

```bash
.venv/bin/python scripts/run_stage6_office.py \
  --seed 15101 --gpu 1 --headless

.venv/bin/python scripts/run_stage6_office.py \
  --seed 25101 --gpu 1 --headless

.venv/bin/python scripts/summarize_stage6.py artifacts/stage6/office

.venv/bin/python scripts/analyze_stage6_diagnostics.py \
  artifacts/stage6/office/matched-seed15101-20261103T154116Z
```

The run writes `summary.json`, `metadata.json`, `comparison.json`, DRL and
Reference trajectory JSONL, separate Ground Truth JSONL, planning history,
`aligned-trajectories.npz`, `trajectory-comparison.png`,
`cost-comparison.png`, resolved configs, and the Isaac Sim log below a unique
`artifacts/stage6/office/matched-seed15101-*` directory.

## 7. Known limitations and remaining acceptance work

- The Office checkpoint is the selected Stage 5 Phase 4 candidate, but Stage 5
  as a whole remains in progress.
- One validation seed and one test seed were evaluated. The same-seed repeat is
  excluded from aggregate statistics. Because resolved pedestrian routes are
  fixed, this is not yet a multi-initialization pedestrian evaluation.
- Hospital and Warehouse integration is deliberately pending their frozen
  independent policies.
- Candidate divergence points are diagnostic candidates, not causal failure
  labels. Counterfactual intervention is not implemented.
- The Gaussian policy distribution and critic value are now exported, but they
  are not calibrated confidence measures and have not undergone causal
  intervention analysis.

Stage 6 can be marked complete only after physical matched evaluations and
dataset validation pass for Office, Hospital, and Warehouse. Stage 7 has not
been started.

## 8. Requirements for completing Stage 6

1. Freeze the independently trained Hospital and Warehouse Stage 5 policies.
2. Add their Stage 6 YAML configurations without sharing Office policy weights.
3. Run matched DRL/reference physical evaluations and same-seed reproducibility
   checks for both environments.
4. Evaluate multiple pedestrian initializations rather than only changing the
   observation-noise seed; this requires separate deterministic resolved route
   sets.
5. Improve or explicitly bound the reference baseline's 34.375% infeasible
   replan rate before interpreting positive Delta J as a policy limitation.
6. Aggregate all three environments and validate privileged/observed-state
   separation in every generated dataset.

Counterfactual causality remains Stage 7 work and is not required to claim only
the current Office integration result.
