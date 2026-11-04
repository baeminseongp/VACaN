# Versioned model checkpoints

Only checkpoints selected for a reproducible experiment belong here. Raw
training runs, periodic snapshots, TensorBoard data, and evaluation artifacts
remain under the ignored `artifacts/` directory.

Binary model files are stored with Git LFS. Before cloning or pushing them:

```bash
git lfs install
git lfs pull
```

The Stage 7 Office experiment uses the frozen SB3-Contrib Recurrent PPO policy
at `stage5/office/recurrent_ppo.zip`. SB3 saves this model as a ZIP archive; it
is the equivalent trained-policy artifact for this project, not a PyTorch
`.pt` file. Future `.pt`, `.pth`, and `.safetensors` files are also covered by
the repository's LFS rules.

Each exported model must have adjacent metadata recording its source artifact,
scenario ownership, seed, training step, SHA-256 digest, and intended use. Do
not copy a checkpoint between Office, Hospital, and Warehouse policies.
