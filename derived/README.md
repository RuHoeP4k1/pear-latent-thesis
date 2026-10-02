# derived/

Ignored by git. Holds intermediate products that are expensive to recompute but can always be rebuilt from `data/` plus code, for example:

- `latents/<encoder>/<run_id>/latents.npy` and `manifest.parquet` — one row per fruit, written by the encoding script on the workstation.
- Preprocessed volumes, if preprocessing is ever cached.

Every product is written together with a `run.json` file (git commit, encoder name, date, parameters) so a result can be traced back to the code that produced it. See `pearlatent.runs`.
