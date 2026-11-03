# Stage 0 — Environment and compatibility verification

Date: 2026-10-30 (Asia/Seoul)
Status: **COMPLETE on GPU 0 — GPU 1 operating policy configured; repeat smoke pending**

## 1. Outcome

The repository targets NVIDIA Isaac Sim, Isaac Lab, Pegasus People, and Nova
Carter with no simulator fallback. The compatible release combination and 5.1
asset locations were verified. On the GPU host, two independent integration
processes loaded the official Warehouse, spawned Nova Carter and a Pegasus
Person, evaluated the animation graph, moved the person 2.828673 m, read robot
and pedestrian positions, and shut down successfully. Both deterministic runs
produced identical initial/final transforms. Stage 0 acceptance tests passed;
Stage 1 was not started.

## 2. Machine inspection

| Item | Actual result |
|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic, x86_64 |
| CPU | 2 × Intel Xeon Silver 4214R; 48 logical CPUs |
| GPU | 2 × RTX 3090 (24 GiB each), 1 × NVIDIA T1000 8GB, from `/proc/driver/nvidia/gpus` |
| Driver | 580.178.04; Isaac Sim host log verifies Vulkan access |
| GPU runtime | Host run detected both RTX 3090s and the T1000; agent sandbox remains isolated |
| CUDA toolkit | `nvcc` absent; `libcuda.so` present |
| RAM | 125 GiB total, 103 GiB available |
| Swap | 8 GiB |
| Storage | 468 GiB filesystem, 340 GiB available |
| Python | system/Conda 3.13.9; `.venv` 3.11.15; `/usr/bin/python3.12` |
| Existing environments | Conda `base`, `raise-vla-train`; project `.venv` |

The missing CUDA compiler is not a blocker for the prebuilt simulator. The host
Isaac Sim log supersedes the coding-agent sandbox result and verifies GPU access.

## 3. Installed software and API evidence

- Isaac Lab source exists at `/home/engineer/IsaacLab`; `VERSION` is `2.3.2`.
- Isaac Sim is linked at `/home/engineer/IsaacLab/_isaac_sim` to
  `/home/engineer/isaacsim-5.1.0`. Its `VERSION` is
  `5.1.0-rc.19+release.26219.9c81211b.gl`, bundled Python is 3.11.13, and
  `import isaacsim` passes.
- Pegasus exists at `/home/engineer/IsaacLab/external/PegasusSimulator`, tag
  `v5.1.0`, commit `644da37e9d5268e5f9a34e78bdcfd57a8bab82b4`.
- Actual People API files are:
  - `extensions/pegasus.simulator/pegasus/simulator/logic/people/person.py`
  - `extensions/pegasus.simulator/pegasus/simulator/logic/people/person_controller.py`
- The installed Pegasus `Person` uses `isaacsim.replicator.agent.core`, registers
  physics callbacks, and updates its target through `update_target_position`.

## 4. Compatibility matrix

| Candidate | Isaac Sim | Isaac Lab | Pegasus | Python / Torch | Decision |
|---|---:|---:|---:|---|---|
| Old iCrowdNav path | 4.0/4.2 | not pinned to installed Lab | vendored 4.2 | older runtime | rejected: conflicts with mandatory installed stack |
| Isaac Lab 2.3 common intersection | **5.1.0** | **2.3.2** | **5.1.0** | bundled Python 3.11 / Torch 2.7.x cu128 | **selected** |
| Newest independently | 6.x | 3.x | 5.1.0 | release-dependent | rejected: Pegasus has no matching verified 6.x release |

Why the selected combination is defensible:

- [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases)
  identify the 2.3 line with Isaac Sim 5.1.
- [Pegasus installation documentation](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html)
  specifies and tests the 5.1.0 standalone simulator.
- The locally installed Isaac Lab 2.3.2 metadata accepts Isaac Sim 4.5, 5.0,
  and 5.1, requires Python 3.10+, and pins its Torch index to cu128 with
  `torch>=2.7`.
- The locally installed Pegasus package and its hard-coded People asset root are
  both version 5.1.0.

This establishes both package-level compatibility and host acceptance on the
installed Ubuntu 24.04 / driver 580.178.04 system.

## 5. Asset verification

The following paths were checked against the official 5.1 content server and
returned HTTP 200 on 2026-10-30:

| Asset | 5.1 relative USD path |
|---|---|
| Office | `/Isaac/Environments/Office/office.usd` |
| Hospital | `/Isaac/Environments/Hospital/hospital.usd` |
| Warehouse | `/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd` |
| Nova Carter | `/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd` |
| Person | `/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd` |

The test resolves these paths against
`isaacsim.storage.native.get_assets_root_path()` at runtime and calls
`omni.client.stat` before spawning. The older
`/Isaac/Robots/Carter/nova_carter.usd` path returned 404 and is not used.
Assets were resolved directly from the official 5.1 content server during both
successful runs; no local-cache-only path is claimed.

## 6. Implemented repository structure

```text
src/vacan/
├── simulation/       # simulator and privileged Ground Truth boundary
├── pedestrians/      # Pegasus Person/PersonController adapters
├── robot/            # Nova Carter and differential-drive adapters
├── sensors/
├── mapping/
├── prediction/
├── rl/
├── reference/
├── counterfactual/
└── analysis/
```

Existing research files and legacy 2D code were preserved. `configs/default.yaml`
now selects `isaac-sim`; the legacy code is not an active fallback.

Reproducibility components include centralized YAML, `seed_everything`, JSON
logging, atomic checkpoint metadata, a pinned runtime manifest, a host evidence
collector, and per-run smoke-test JSON.

Created for this Stage 0 pass:

- `configs/isaacsim/stage0_smoke.yaml`
- `requirements/stage0-isaac-manifest.yaml`
- `scripts/check_stage0_host.py`
- `scripts/isaacsim/stage0_integration_smoke.py`
- `scripts/run_stage0_smoke.sh`
- `src/vacan/pedestrians/__init__.py`
- `src/vacan/robot/__init__.py`

Updated: `AGENTS.md` (already contains the supplied instructions), `README.md`,
`configs/default.yaml`, `docs/architecture.md`, `docs/stage-0-report.md`,
`docs/third-party.md`, the host-requirement comments, and Stage 0 tests.

## 7. Minimal integration test

`scripts/isaacsim/stage0_integration_smoke.py` starts the Pegasus-compatible
Isaac Sim base experience directly with `SimulationApp` and uses Isaac Lab
`UsdFileCfg` for scene assets. It creates the fresh USD stage required by the
official Pegasus People example, creates the Pegasus `World`, loads an official
environment, spawns Nova Carter, and creates one `Person` with a concrete
`PersonController`. Before stepping, it verifies that the character SkelRoot,
Biped animation graph, and `AnimationGraphAPI` all exist on the active World
stage. After reset it starts the timeline, performs five render-backed
USD/Animation Graph/Fabric synchronization steps, and verifies that
`omni.anim.graph.core.get_character()` returns a live handle. This post-reset
warm-up—not a change of headless mode—was the critical final correction. It
then steps 240 frames, reads both transforms, requires at least 0.05 m person
movement, writes a result, and closes `SimulationApp` in `finally`.

`scripts/run_stage0_smoke.sh` starts it in two separate processes so clean
shutdown is tested rather than inferred from two resets in one process.

## 8. Tests and actual results

| Check | Result |
|---|---|
| Host evidence collector | PASS as a diagnostic; sandbox GPU isolation is superseded by host smoke evidence |
| YAML/repository unit tests | PASS: 18 passed in 1.34 s after final correction |
| Isaac Sim Python import | PASS: installed 5.1.0 runtime imports |
| Pegasus editable install | PASS: version 5.1.0 in Isaac Sim Python |
| Isaac Lab import in Isaac Python | PASS: installed modules import from Isaac Sim Python |
| Isaac Sim GPU launch | PASS: driver 580.178.04, Vulkan, 2 × RTX 3090 + T1000 detected |
| Isaac Lab Tasks extension | PASS after installing `h5py==3.14.0` |
| Pegasus Person creation | PASS: valid `SkelRoot`, Biped graph, and applied `AnimationGraphAPI` |
| Runtime interpreter selection | Corrected: launcher now invokes Isaac Sim `python.sh` directly instead of Conda-sensitive `isaaclab.sh -p` |
| Historical failed runs | Resolved: stage/extension ordering, interpreter selection, GPU selection, and post-reset Fabric warm-up corrected |
| Corrected launcher static checks | PASS: official fresh-stage order restored, duplicate bundles removed, pre-step animation validation added |
| Animation Graph activation | PASS after five post-reset render/Fabric synchronization steps and explicit live-handle verification |
| Official Warehouse USD load | PASS in both runs |
| Nova Carter spawn/transform | PASS: `[0, 0, 0]` read in both runs |
| Pegasus person spawn/movement/transform | PASS: 2.828672975 m movement in both runs |
| Coexistence and two clean exits | PASS: two independent result JSON files report `pass` |

Generated evidence: `artifacts/stage0/host-inspection.json`,
`smoke-run-1.json`, `smoke-run-2.json`, and their corresponding logs under
`artifacts/stage0/`.

The measured in-test elapsed times were 25.699 s and 22.530 s; total process
lifetimes visible in the logs were about 44.6 s and 40.7 s. Both 240-step runs
produced exactly the same person transforms and displacement for seed 20261030.
Those accepted results used GPU 0. The requested operational policy now defaults
to GPU 1 (the second RTX 3090) in visible mode, expects Xorg on
`PCI:115:0:0`, and rejects GPU 2 (T1000); a repeat run is required before
claiming GPU 1-specific acceptance evidence.

## 9. Exact commands

Inspect the host:

```bash
.venv/bin/python scripts/check_stage0_host.py \
  --output artifacts/stage0/host-inspection.json
```

Run or repeat the accepted GPU smoke test:

```bash
bash scripts/run_stage0_smoke.sh
```

The runner selects GPU 1 in GUI mode by default. Use `VACAN_HEADLESS=1` for
compute-only execution and set `VACAN_GPU_INDEX=0` only when intentionally
using the other RTX 3090. GPU 2 is rejected because it lacks Isaac Sim support.

Large-download and installation commands are in the root README. The selected
dependency record is `requirements/stage0-isaac-manifest.yaml`.

## 10. Known limitations and requirements for Stage 1

1. The accepted non-headless run selected GPU 0 (RTX 3090), but Xorg is
   configured for `PCI:166:0:0`, the T1000 at Isaac Sim GPU index 2. Vulkan
   therefore reported that GPU 0 had no presenting queue, `createSwapchain`
   failed, and subsequent render ticks logged that backbuffers were not
   initialized. State evaluation still passed because GPU compute/render graph
   execution does not require window presentation. The T1000 is not an Isaac
   Sim alternative: its 8 GB VRAM is below the 5.1 minimum and NVIDIA's product
   matrix lists no RT Cores. VACaN now expects Xorg and Isaac Sim on GPU 1,
   rejects GPU 2, and supports an explicit headless override for compute-only
   runs. GPU 1 GUI acceptance remains pending until the display/Xorg migration
   and repeated smoke test complete.
2. Pegasus eagerly probes its optional ROS2 backend, so ROS2 Bridge warnings
   appear when ROS2 is absent. Stage 0 does not use ROS2.
3. The minimal acceptance test loads Warehouse only. Office and Hospital asset
   URLs were verified but full scenario construction belongs to Stage 1.
4. Preserve the privileged Ground Truth/observed-state boundary when building
   scenario state recording and sensor interfaces.

Stage 1 has deliberately not been started automatically.
