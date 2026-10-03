# Move the SAE experiment to GitHub and a Windows PC

The source repository and the collected data are different things. Git should contain the code, configuration, fixtures, tests, methods, and a compact copy of the findings. The existing `runs/` and `cache/` trees are ignored by Git; most run directories and `.venv` are Mac-only symlinks. Cloning the code alone can run software tests and show the [portable report](../reports/sae_analysis_v1/report.html), but cannot recompute the published SAE findings until the collected data are copied too.

## 1. Create a dedicated private GitHub repository

On GitHub, create an **empty private** repository named, for example, `quality-dimensions-llm`; do not initialize it with a README. In Terminal on the Mac:

```bash
cd "/Users/popthornthanatit/Desktop/CS/diss/qualityDims/quality-dimensions-llm"
git init -b main
git rev-parse --show-toplevel
git add .gitignore README.md pyproject.toml requirements-lock.txt src tests configs data docs analysis handoff populate_quality_suite.py reports/sae_analysis_v1
git diff --cached --stat
git status --short
git commit -m "Add reproducible quality-dimensions experiment"
git remote add origin https://github.com/YOUR_USERNAME/quality-dimensions-llm.git
git push -u origin main
```

The `git rev-parse` output **must** be the project directory above before `git add`. Until `git init` is run there, this folder inherits an unrelated Git repository rooted at `/Users/popthornthanatit`, with a university remote. Review the staged file list before committing and keep credentials, `.venv`, `runs/`, and `cache/` out of Git. Replace `YOUR_USERNAME` with the account that owns the new repository. For HTTPS push authentication, use Git Credential Manager/GitHub CLI or a personal access token when prompted; [GitHub account passwords do not work for Git pushes](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).

The committed report copy is [here](../reports/sae_analysis_v1/report.html), with [findings](../reports/sae_analysis_v1/findings.md) and [SVG figures](../reports/sae_analysis_v1/plots/). The interactive explorer and per-record outputs stay in the full analysis run.

## 2. Connect the repository to ChatGPT

In ChatGPT/Codex, choose **Work in → Cloud → Select environment → Create environment**, connect GitHub when prompted, select this private repository, review the proposed setup, and publish the environment. This gives Codex cloud tasks access to the repository; see the [OpenAI Docs cloud quickstart](https://learn.chatgpt.com/docs/cloud). For ordinary repository search in ChatGPT, install/connect the GitHub integration from the Plugins directory when available and authorize this repository; see [OpenAI Docs plugins](https://learn.chatgpt.com/docs/plugins). Availability can differ between ChatGPT surfaces. Repository access only includes what you committed; it does not upload your Mac's ignored runs or caches.

## 3. Copy the existing collected data separately

For a rerun of this analysis **without extracting model activations again**, copy these three run directories and the pinned V3 cache to an external SSD or another large-file transfer service. The transfer needs roughly 3 GB before any new outputs or model checkpoint downloads:

| Folder | Approximate size | Purpose |
|---|---:|---|
| `runs/sae_scope_v1_laptop_top100-de55e3b9d113` | 2.2 GB | Raw residuals, SAE codes, metadata and manifest |
| `runs/quality_suite_v3_laptop-88fad55b46b0` | 66 MB | Saved V3 behaviour scores and source metadata |
| V3 activation cache named in the collection's `source_v3.json` | 443 MB | Residuals reused from the earlier V3 experiment |
| `runs/sae_analysis_v1_core-5fe3b13cdf8a` | 431 MB | Optional: the already computed full results and explorer |

For an external drive mounted on the Mac as `/Volumes/TRANSFER`, this copies the **symlink targets** into ordinary folders:

```bash
cd "/Users/popthornthanatit/Desktop/CS/diss/qualityDims/quality-dimensions-llm"
TRANSFER="/Volumes/TRANSFER/qualitydims-transfer"
mkdir -p "$TRANSFER/runs" "$TRANSFER/cache/quality_suite_v3"
for name in sae_scope_v1_laptop_top100-de55e3b9d113 quality_suite_v3_laptop-88fad55b46b0 sae_analysis_v1_core-5fe3b13cdf8a; do
  rsync -aL "runs/$name/" "$TRANSFER/runs/$name/"
done
CACHE_SRC=$(python3 -c 'import json; print(json.load(open("runs/sae_scope_v1_laptop_top100-de55e3b9d113/source_v3.json"))["activation_cache"]["root"])')
rsync -a "$CACHE_SRC/" "$TRANSFER/cache/quality_suite_v3/$(basename "$CACHE_SRC")/"
```

If you only want to recompute the analysis, omit `sae_analysis_v1_core-5fe3b13cdf8a` from the `for name` line. The Mac currently has little free internal storage, so create this transfer on the external drive rather than making another local 3 GB copy. Standard GitHub is a poor place for the binary collection; [GitHub recommends small repositories and blocks regular Git files over 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). [Git LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs) or a [separate large-file release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) can be used if you explicitly want GitHub-hosted data, but account for storage/bandwidth and ensure the PC receives real data files rather than LFS pointers.

## 4. Set up the PC and restore the data

Install Git and Python 3.12 on Windows. In PowerShell, replace `YOUR_USERNAME` and drive `E:`:

```powershell
git clone https://github.com/YOUR_USERNAME/quality-dimensions-llm.git
cd quality-dimensions-llm
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pytest -m "not integration" -q
robocopy "E:\qualitydims-transfer\runs" ".\runs" /E
robocopy "E:\qualitydims-transfer\cache" ".\cache" /E
```

`robocopy` may return a nonzero success code for copied files. Confirm the collection's `manifest.json` and the V3 cache's `identity.json` exist after copying. The Mac's `requirements-lock.txt` pins an ARM64 environment, so use the project dependencies for the first Windows install and record a Windows-specific lock after the tests pass. The copied source metadata retains the original Mac paths as provenance; the analysis now resolves the same V3 run and cache from the local `runs/` and `cache/` folders when those original paths do not exist. Environment variables `QD_V3_RUN_DIR` and `QD_V3_CACHE_BASE` are available if you place those folders elsewhere.

Then run the bounded analysis on the PC. Explicit output folders avoid Windows symlink privileges:

```powershell
.\.venv\Scripts\qd.exe sae-analyze run --profile smoke --run-dir .\runs\pc-sae-smoke
.\.venv\Scripts\qd.exe sae-analyze run --profile core --run-dir .\runs\pc-sae-core
```

The core run records its source/configuration/code hashes in `identity.json`; use `runs\pc-sae-core` for the optional `enrich`, `optional-methods`, `context`, `reconstruction`, `baselines`, and `report` commands in [the analysis methods](SAE_ANALYSIS_V1.md). Choose a new output folder if you change the configuration or source code, and use `--resume` to continue an interrupted run in the same folder. `qd sae-analyze report` writes `findings.md`, figures and an offline `explorer.html`; the additional portable `report.html` packaging command in the methods document points to the Mac's installed builder, so use the tracked report copy on Windows unless that builder is also available there. The core result is compute-intensive even without rerunning Qwen. Exact decoder diagnostics and dictionary PCA also need the nine pinned Qwen-Scope SAE checkpoint files (about 4.8 GB total) in the Hugging Face cache; otherwise those specific checks record that the checkpoints are unavailable. Re-extracting the original model activations from scratch is a separate, much larger task and the saved V3 configuration specifies Mac MPS, so a Windows GPU/CPU reproduction requires a new backend configuration and will get new run identities.
