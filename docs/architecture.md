# VACaN architecture and access boundary

```text
Isaac Sim official environment ─┬─ Nova Carter sensors ─> observed state
Pegasus PersonController ───────┘          │
                                            ├─ congestion estimate
                                            ├─ intention prediction
                                            └─ timestamp freshness
                                                        │
                                               Recurrent PPO (Isaac Lab)
                                                        │
                                        differential drive (v, omega)

Isaac Sim complete state ─> privileged Ground Truth interface
                              ├─ training-label generation
                              ├─ reference planning
                              ├─ evaluation
                              └─ counterfactual analysis
```

The observed-state policy interface and privileged simulator interface are
separate modules. Ground Truth state, global pedestrian poses, exact occupancy,
and simulator handles must never be passed to the navigation policy.

CCTV-covered grid cells may expose exact current crowd density as the explicit
controlled research assumption in `AGENTS.md`; those cells have no freshness
decay. Robot observations update visible non-CCTV cells, whose age remains an
independent Freshness channel. Congestion, Intention, and Freshness are logged
as independently identifiable factors.

Stage 0 establishes only the runtime and package boundaries. Prediction models,
Recurrent PPO training, reference planning, and counterfactual experiments are
not implemented in this stage.

The preserved reconstructed 2D simulator is legacy research code. It is not a
fallback for Isaac Sim and is not selected by `configs/default.yaml`.
