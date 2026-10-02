# Workflow across machines

## 1. The three machines

| Machine | Operating system | Role | PyTorch build |
|---|---|---|---|
| Laptop | Windows | Main editing, Claude Code sessions, statistics on saved latents, figures | CPU |
| School PC | Windows, no administrator rights | Same as laptop when away from it | CPU |
| Lab workstation | Linux (to confirm), NVIDIA GPU, reached by SSH | Encoding CT volumes, fine-tuning | CUDA 12.8 |

Git is the only route by which code moves between machines. Data moves separately (university
storage, `scp`), and never through git or GitHub.

## 2. First-time setup on a machine

### Windows (laptop or school PC), no administrator rights needed

```powershell
# 1. Install uv into your user folder
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 2. Get the code
git clone git@github.com:<your-user>/pear-latent-thesis.git
cd pear-latent-thesis
# 3. Install Python 3.12 and every package, exactly as in uv.lock
uv sync
# 4. Machine-specific paths
copy config\local.example.toml config\local.toml
notepad config\local.toml
# 5. Check the environment
uv run marimo edit notebooks\00_environment_check.py
```

Or run `scripts\setup.ps1`, which does steps 1, 3 and 4.

If git is not installed on the school PC and cannot be installed, use the portable version
("PortableGit" from git-scm.com), which runs from the user folder.

Claude Code on the school PC: install it in the user folder with the native installer from the
Claude Code documentation, and sign in. If installation is not allowed, edit code there and run
Claude Code only on the laptop.

### Lab workstation (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:<your-user>/pear-latent-thesis.git
cd pear-latent-thesis
bash scripts/setup.sh        # uv sync, local config, GPU check
```

## 3. Daily cycle

1. `git pull` and `uv sync` (picks up any package change another machine committed).
2. Start Claude Code in the repository folder. The session-start hook shows the current state from
   `HANDOFF.md`.
3. Open marimo next to it: `uv run marimo edit --watch notebooks/NN_name.py`. With `--watch` the
   editor reloads when Claude edits the file.
   For live collaboration where Claude runs cells in your open notebook, install the official
   `marimo-pair` skill (section 6) and type `/marimo-pair pair with me on notebooks/NN_name.py`.
4. Work. Commit small steps.
5. `/handoff`, then commit and `git push`.

## 4. PyTorch and CUDA

`pyproject.toml` sends Linux to the `cu128` PyTorch index and Windows to the CPU index. The CUDA
wheel works only if the workstation's NVIDIA driver supports that CUDA version. Check with
`nvidia-smi`: the "CUDA Version" in its header is the highest version the driver supports.

- 12.8 or higher: keep `cu128`.
- 12.6 to 12.7: change both `pytorch-cu128` entries to `cu126` (URL and name), then `uv lock`.
- Lower: ask the workstation administrator about a driver update, or use `cu118`.

If your laptop has an NVIDIA GPU and you want CUDA there too, change the Windows marker from
`pytorch-cpu` to the same CUDA index. Commit `uv.lock` after every change.

## 5. Data and the cloud (molab)

molab gives free GPU notebooks, but molab notebooks are public (unlisted) by default and data
uploaded there leaves the university. Rule until Hugo confirms otherwise in writing: molab only for
toy data or public examples. The project workflow does not depend on it.

## 6. Optional: official marimo agent skills

The marimo team publishes skills that teach agents marimo itself (`marimo-notebook`) and let
Claude Code operate a running notebook (`marimo-pair`). They need Node.js (`npx`) or uv:

```bash
npx skills add marimo-team/skills --agent claude-code
npx skills add marimo-team/marimo-pair --agent claude-code
# without Node.js:
uvx deno -A npm:skills add marimo-team/marimo-pair
```

They install into `.claude/skills/`. Commit them so every machine has the same version. The
repository's own `thesis-notebook` skill takes precedence where the two disagree.

## 7. Remote marimo on the workstation

See `.claude/skills/workstation-run/SKILL.md`. Short version: start
`uv run marimo edit --headless --host 127.0.0.1 --port 8080` on the workstation, forward with
`ssh -N -L 3718:127.0.0.1:8080 USER@WORKSTATION`, open http://localhost:3718. VS Code users can
instead use the Remote-SSH extension, which forwards the port automatically.
