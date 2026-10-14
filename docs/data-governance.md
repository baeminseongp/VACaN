# Data governance and provenance

## Lifecycle

1. `external`: immutable source archives and license snapshots.
2. `interim`: extracted or partially transformed files.
3. `processed`: canonical records consumed by experiments and the UI.
4. `synthetic`: small committed fixtures or generated controlled scenarios.

Large data is ignored by Git. Every future dataset adapter must record source
URL, retrieval date, checksum, license/access terms, transformation version,
and scenario ID in a manifest.

## Privacy defaults

- Prefer trajectories and derived counts over identifiable imagery.
- Preserve identity only when required by an approved evaluation protocol.
- Do not infer sensitive personal attributes.
- Anonymize selected frames before sending them to an external VLM.
- Separate observable evidence from model-generated explanations.

## Missingness

Unknown density is stored as missing, never as zero. `last_observed` and
`observation_age_s` express information freshness. This distinction must
survive ingestion, API serialization, analysis, and visualization.
