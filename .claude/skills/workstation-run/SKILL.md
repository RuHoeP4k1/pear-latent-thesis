---
name: workstation-run
description: How to prepare, launch and collect long or GPU jobs on the MeBioS lab workstation over SSH (encoding CT volumes, fine-tuning). Load before writing anything that will run on the workstation or when the user asks to run something on the GPU.
---

# Running work on the lab workstation

The workstation is reached by SSH. Code arrives there only through git; data is already there.

## Before launching

1. Commit and push the code from the laptop. On the workstation: `git pull`, then `uv sync`.
   A run from uncommitted code is not reproducible; `run.json` records `dirty: true` if so.
2. Check the GPU is free: `nvidia-smi`. If another user is on it, say so and stop.
3. Run on a tiny subset first (`--limit 4`) and confirm output shapes before the full run.

## The script path of a notebook

Long jobs are marimo notebooks run as scripts. The notebook reads command-line arguments with
`mo.cli_args()`, which works both as a script (`uv run python nb.py --limit 4`) and in the editor
(`uv run marimo edit nb.py -- --limit 4`). Values arrive as strings, so convert them:

```python
@app.cell
def _():
    encoder_name = mo.cli_args().get("encoder") or "hugo_vae"
    limit = int(mo.cli_args().get("limit") or 0)  # 0 means all volumes
    return encoder_name, limit
```

The heavy cell is gated so it waits for the button in the editor but runs directly as a script:

```python
mo.stop(mo.running_in_notebook() and not run.value, mo.md("_Press the button to start._"))
```

(Tested with marimo 0.25.1: in script mode `mo.running_in_notebook()` is False and
`mo.app_meta().mode` is `"script"`.)

Launch inside tmux so the job survives a dropped SSH connection:

```bash
tmux new -s encode
uv run python notebooks/02_encode_latents.py --encoder hugo_vae 2>&1 | tee results/encode.log
# detach: Ctrl-b then d      reattach later: tmux attach -t encode
```

## Interactive marimo on the workstation

On the workstation:

```bash
uv run marimo edit --headless --host 127.0.0.1 --port 8080 notebooks/
```

On the laptop (PowerShell):

```powershell
ssh -N -L 3718:127.0.0.1:8080 USER@WORKSTATION
```

Then open http://localhost:3718. Use a port other than marimo's default 2718 on the workstation,
so it does not clash with a marimo already running on the laptop. Keep the access token marimo
prints; do not pass `--no-token` on a shared machine.

## Outputs

1. Everything goes to `derived/` or `results/<run_id>/` created by `pearlatent.runs.new_run_dir`,
   with `write_run_record(run_dir, params)` called before the job starts.
2. Small products needed on the laptop (latents for ~1,100 fruit are small: 32³ floats × 1,100 ≈
   144 MB as float32) are copied with `scp` or placed on university storage — never committed.
3. After the run, append to `HANDOFF.md`: run id, what was run, where the output is.

## Failure handling

- CUDA out of memory: lower `encode_batch_size` in `config/local.toml`, do not change code.
- `torch.cuda.is_available()` is False although a GPU exists: the CUDA wheel does not match the
  driver. Compare `nvidia-smi` (driver's maximum CUDA version) with the `pytorch-cu1xx` index in
  `pyproject.toml`; see `docs/workflow.md` section 4.
