# Stage 7 Office run archive

This directory is a byte-for-byte archive of every retained Stage 7 Office run
from the development workspace at packaging time (2026-11-04). It contains six
closed-loop runs plus the standalone host-side `decision/` replay output.

All six runs used seed `15101` and report `status: pass` at the launcher level:

- `run-seed15101-20261103T160240Z/`
- `run-seed15101-20261104T052612Z/`
- `run-seed15101-20261104T061342Z/`
- `run-seed15101-20261104T102921Z/`
- `run-seed15101-20261104T115353Z/`
- `run-seed15101-20261104T122140Z/`

Each run preserves its resolved YAML, commands, logs, per-branch JSONL
trajectories, intervention records, NPZ arrays, numerical results, and rendered
figures. The complete Stage 7 archive is stored as ordinary Git files so it is
available immediately after cloning without an additional LFS download.

The timestamped directories are successive experiment designs, not independent
evaluation seeds. Use the latest run for the current Case 1-2 and Case 2 design,
and retain the earlier directories only as provenance for how the case studies
changed. A launcher-level pass means the configured run completed and its
structural checks passed; it does not imply that every injected error increased
navigation cost. In the latest run, Case 1-2 and Case 2 have negative
`delta_j_comparison_minus_correct`, so they remain negative scientific results
and must not be presented as confirmed inefficient-path failures.

Absolute paths in recorded commands and summaries describe the originating
machine. Use the repository commands and local paths for a new reproduction.
