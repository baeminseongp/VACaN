# Architecture

## Design objective

VACaN supports reproducible component validation and an interactive visual
analytics system without coupling research algorithms to a specific dataset or
frontend framework.

```text
dataset adapters / simulation
             |
             v
 canonical spatiotemporal contracts
             |
      +------+------+
      |             |
      v             v
 analytics      API / web UI
 density        map + timeline
 coverage       coverage + paths
 planning       provenance views
      |             |
      +------+------+
             v
 experiment metrics and PacificVis artifacts
```

## Boundaries

- `src/vacan`: deterministic, dataset-independent domain logic.
- `experiments`: scenario runners, evaluation protocols, and generated metrics.
- `apps/api`: delivery layer for canonical records and derived analytics.
- `apps/web`: linked visual views; it does not become the source of research
  truth.
- `paper/pacificvis`: paper-only manifests and publication artifacts. Generated
  figures should be traceable to an experiment configuration.
- `data`: local data lifecycle. Licensing may prohibit committing source data.

## Scenario isolation

ATC, CAVIAR, JRDB, Mall, and synthetic scenes are independent scenarios. A
future integrated synthetic scene may combine roles, but real datasets must not
be represented as if they were synchronized sensors in one physical venue.

## Time and coordinates

- Timestamps use ISO 8601 UTC in canonical files.
- Metric world coordinates use meters.
- Each scenario declares a coordinate reference and map origin.
- Video-only data retains frame/time-code identifiers in adapter-specific
  provenance before projection into world coordinates.
