# VACaN

VACaN is a research-oriented prototype for visual analytics of indoor crowd
sensing, spatial congestion, observation coverage, and multi-robot navigation.

The repository separates reproducible research code from the interactive
system and PacificVis-facing artifacts:

```text
VACaN/
├── apps/                    # Deployable API and visual-analytics clients
│   ├── api/
│   └── web/
├── configs/scenarios/       # Versioned experiment/scenario configuration
├── data/                    # Data lifecycle; large source data is never committed
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── synthetic/
├── docs/                    # Architecture and data-governance decisions
├── experiments/             # Reproducible experiment entry points and outputs
├── paper/pacificvis/        # Paper-specific figures, tables, and notes
├── schemas/                 # Canonical cross-component data contracts
├── src/vacan/               # Dataset-independent research logic
└── tests/                   # Contract and unit tests
```

Dataset storage is configured separately in
[`configs/storage.json`](configs/storage.json); public and generated datasets
are separated. Public source datasets can be read-only under
`/media/engineer/DATA1/datasets`, while generated derivatives are written
to the ignored project directory `data/processed`.

## Current scope

Step 1 defines the repository architecture, canonical data contracts, the
observation-freshness state model, and a tiny synthetic scenario. Step 2 adds
the research-only ATC adapter and its offline fixture. The repository does not
redistribute ATC, CAVIAR, JRDB, or Mall source data.

## Quick check

Python 3.11 or newer is required. No third-party package is needed for the
Step 1 validation suite.

```bash
python -m unittest discover -s tests -v
python -m vacan.validation data/synthetic/minimal
```

When running the second command without installing the package, set
`PYTHONPATH=src` or use `make validate`.

ATC acquisition, normalization, and alignment instructions are documented in
[`docs/datasets/atc.md`](docs/datasets/atc.md).
CAVIAR source provenance and Step 6 input boundaries are documented in
[`docs/datasets/caviar.md`](docs/datasets/caviar.md).
Step 3 congestion and observation-state decisions are documented in
[`docs/analytics/step3.md`](docs/analytics/step3.md).
Step 4 route comparison and replanning decisions are documented in
[`docs/planning/step4.md`](docs/planning/step4.md).
Step 5 linked-view dashboard and privacy-preserving export decisions are
documented in [`docs/visualization/step5.md`](docs/visualization/step5.md).
Step 6 CAVIAR real-video projection and zone-count validation decisions are
documented in [`docs/evaluation/step6.md`](docs/evaluation/step6.md).
Step 7 keeps JRDB mobile sensing and Mall dense-crowd/VLM evaluation independent;
its contracts and licensed-data boundary are documented in
[`docs/evaluation/step7.md`](docs/evaluation/step7.md).
The early ATC system evaluation has been reclassified as a Step 8 foundation
in [`docs/evaluation/step8-foundation.md`](docs/evaluation/step8-foundation.md).

## Core modeling rules

- Continuous density is retained internally; Low/Medium/High is a view-layer
  classification.
- `UNKNOWN` is not zero congestion. Unknown density is serialized as an empty
  CSV value.
- Every derived observation carries its source, confidence, and observation
  age.
- Public datasets remain separate evaluation scenarios. They are never
  presented as simultaneous sensors in one real venue.
