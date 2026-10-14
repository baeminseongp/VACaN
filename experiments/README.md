# Reproducible experiments

Each experiment will declare a scenario, configuration, code revision, random
seed, metrics, and output manifest. Dataset component validation and integrated
synthetic evaluation remain separate experiment families.

The ATC paired evaluation is retained as a Step 8 foundation:

```bash
make atc-evaluation
```

It writes paired sensing and planning records, timestamp-block bootstrap
intervals, input hashes, and paper-ready SVG/Markdown artifacts.

The real-video Step 6 evaluation is run separately with
`make caviar-evaluation` so that ATC and CAVIAR are never represented as
synchronized sensors from one venue.
