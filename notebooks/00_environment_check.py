import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="00 Environment check")

with app.setup:
    # Imports shared by every cell. Keep this block import-only.
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from pearlatent.device import describe_device, seed_everything


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 00 Environment check

    Run this notebook once on every machine (laptop, school PC, lab workstation)
    after `uv sync`. It confirms that marimo, Polars and PyTorch work, shows which
    device PyTorch uses, and demonstrates the conventions every thesis notebook
    follows:

    1. imports in the setup block only;
    2. user interface elements in their own cell, their values read in another cell;
    3. expensive work behind a run button with `mo.stop`;
    4. temporary variables prefixed with an underscore so they do not leak.
    """)
    return


@app.cell
def _():
    seed_everything(0)
    device_info = describe_device()
    mo.ui.table(
        pl.DataFrame([device_info.__dict__]).transpose(
            include_header=True, header_name="property", column_names=["value"]
        ),
        selection=None,
        label="PyTorch device",
    )
    return (device_info,)


@app.cell
def _(device_info):
    mo.callout(
        mo.md(
            f"PyTorch uses **{device_info.device}**. "
            + (
                "GPU work can run on this machine."
                if device_info.device == "cuda"
                else "No CUDA GPU here: run encoding and fine-tuning on the lab workstation."
            )
        ),
        kind="success" if device_info.device == "cuda" else "info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reactive example: a Polars table driven by a slider
    """)
    return


@app.cell
def _():
    n_fruit = mo.ui.slider(20, 500, value=120, step=20, label="Number of simulated fruit")
    noise = mo.ui.slider(0.0, 2.0, value=0.5, step=0.1, label="Noise standard deviation")
    mo.hstack([n_fruit, noise], justify="start")
    return n_fruit, noise


@app.cell
def _(n_fruit, noise):
    _rng = np.random.default_rng(0)
    _grade = _rng.integers(0, 4, n_fruit.value)
    simulated = pl.DataFrame(
        {
            "box": _rng.choice(list("ABCDEFGH"), n_fruit.value),
            "browning": _grade,
            "latent_score": _grade + _rng.normal(0, noise.value, n_fruit.value),
        }
    )
    return (simulated,)


@app.cell
def _(simulated):
    _chart = (
        alt.Chart(simulated)
        .mark_boxplot()
        .encode(
            x=alt.X("browning:O", title="Browning grade (simulated)"),
            y=alt.Y("latent_score:Q", title="Score along one latent direction"),
        )
        .properties(height=260)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Expensive work behind a button
    """)
    return


@app.cell
def _():
    run_benchmark = mo.ui.run_button(label="Run 4096 x 4096 matrix multiplication")
    run_benchmark
    return (run_benchmark,)


@app.cell
def _(device_info, run_benchmark):
    mo.stop(not run_benchmark.value, mo.md("_Press the button to run the benchmark._"))

    def _benchmark() -> float:
        # Tensors live inside a function so GPU memory is released afterwards
        import torch

        _a = torch.randn(4096, 4096, device=device_info.device)
        _b = torch.randn(4096, 4096, device=device_info.device)
        if device_info.device == "cuda":
            torch.cuda.synchronize()
        _start = time.perf_counter()
        for _ in range(10):
            _a @ _b
        if device_info.device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - _start) / 10

    _seconds = _benchmark()
    mo.md(f"One multiplication took **{_seconds * 1000:.1f} ms** on {device_info.name}.")
    return


if __name__ == "__main__":
    app.run()
