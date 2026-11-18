# aa_pipeline_v2: replication package

Code and data for:

> Bhattacharya, A., Gangopadhyay, S., Gopi, P., Ramadas, K., Pandya, P., and Soni, V. *Stylometric Detection of LLM Literary Imitation: A Geometric Analysis Across Ten Canonical Authors.* Digital Scholarship in the Humanities (under review).

Four large language models were prompted to imitate ten nineteenth- and early twentieth-century British prose authors (2,000 imitations), and also wrote 500 non-imitative controls. All texts were scored with Burrows' Delta, Cosine Delta and a PCA centroid diagnostic in three feature spaces. Every result was calibrated against authentic texts using leave-one-out, held-out anchors, and leave-one-work-out validation.

## Contents

| Folder | What it holds |
|---|---|
| `1_standardized_books/` | The 48 Project Gutenberg works (10 authors), cleaned of Gutenberg headers, footers, contents and chapter headings |
| `2_chunked_books/` | All 5,495 chunks of 1,000 words, one folder per author, plus `chunk_manifest.csv` |
| `3_sampled_chunked_books_140/` | The reference set: 140 chunks per author (1,400), plus `chunk_manifest.csv` |
| `anchors/` | The 250 anchor chunks (25 per author), `anchor_manifest.csv`, and `spare_manifest.csv` (the 3,845 spare chunks used to top up LOWO folds) |
| `cues/` | `cue_bank.json`: the four-sentence author cue used in every imitation prompt; `audit_cues.py` |
| `scenarios/` | `scenarios_final.csv` (500 scenarios), `scenarios_final_paired.csv` (with anchor pairings), the drafts, `selection_log.csv`, the automatic check scripts and their reports, and a copy of the standardized books used by the checks (`books/`) |
| `prompts/` | All 1,000 prompts (`imitation/`, 500; `control/`, 500), `call_manifest.csv` (the 2,500 calls with their seeds), `control_key.csv`, and `build_prompts.py` |
| `generation/` | `runner.py` (OpenRouter runner), `models.json` (frozen models, pinned providers, decoding settings), and `outputs/` (see below) |
| `aav2/` | The analysis pipeline (chunking, feature extraction, reference spaces, validation, scoring, statistics) |
| `robustness/` | Four additional scripts: saved reference spaces, PC1/PC2 coordinates, the mixed-effects model, and the copy check |
| `out/` | All pipeline outputs: feature tables, LOO/LOWO/threshold files, per-text scores (`generated.csv`), `results/` (every table in the paper and supplement), and `space/` (fitted reference spaces) |

### Generated texts

- `generation/outputs/main/texts/`: the 2,500 analysed texts (2,000 imitations named `{AUTHOR}_{NN}__{MODEL}.txt` and 500 controls named `C{NNN}__{MODEL}.txt`).
- `generation/outputs/main/calls.jsonl`: one record per API attempt (request, response, seed, served provider, timestamp, word count, acceptance check).
- `generation/outputs/short_originals_replaced/`: six M1 texts that were still under 800 words after their one permitted retry. They were regenerated in a separate run with the same parameters, and the replacements are in `main/texts/`. That run is not recorded in `calls.jsonl`.
- `generation/outputs/pilot/`: a 32-call test of the runner (25 October 2026), made before the main run. It is not analysed in the paper, and it is not the pilot study described in Section 2.2 of the paper.

Control prompts are blinded: `prompts/control/C###.md` carries no author or scenario, and the link back is in `prompts/control_key.csv`.

## Models and settings

| Slot | Model (OpenRouter id) | Pinned provider |
|---|---|---|
| M1 | `xiaomi/mimo-v2.6-pro` | Novita, fp8 |
| M2 | `qwen/qwen3.8-27b` | Parasail, fp8 |
| M3 | `z-ai/glm-5.2` | Novita, fp8 |
| M4 | `deepseek/deepseek-v4.1-flash-1017` | Nebius, fp8 |

All calls used temperature 0.8, top-p 0.95, top-k 50, at most 2,500 output tokens, reasoning disabled, a single user message with no system prompt, provider fallback disabled, and a per-call seed from `prompts/call_manifest.csv`. They were made on 25–26 October 2026.

## Reproducing the analysis

Requires Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin/pip on macOS/Linux
.venv/Scripts/python -m aav2 all
```

`python -m aav2 <stage>` runs a single stage. The stages, in order:

| Stage | What it does |
|---|---|
| `chunk` | (optional) segment the 48 standardized books into 1,000-word chunks |
| `prepare` | assemble documents, truncate generated texts to 1,000 words, plan the LOWO folds |
| `extract` | one spaCy parse per document; fixed features and token/POS cache |
| `validate` | authentic baselines: LOO, thresholds, anchors, LOWO |
| `score` | the 2,000 imitations and 500 controls |
| `analyze` | all research-question tables, controls, and MFW sensitivity |

Results are written to `out/` (use `--out DIR` to write elsewhere). The robustness scripts run afterwards, from the repository root:

```bash
.venv/Scripts/python robustness/save_space.py    # -> out/space/
.venv/Scripts/python robustness/pca_coords.py    # -> out/results/pca_coords.csv, pca_variance.csv
.venv/Scripts/python robustness/glmm.py          # -> out/results/rq2_glmm*.csv
.venv/Scripts/python robustness/copy_check.py    # -> out/results/copy_check_*.csv
```

Each script's docstring describes its method and outputs in full.

## Rebuilding the inputs (optional)

The generation inputs can be rebuilt with the scripts in `scenarios/`, `cues/` and `prompts/`. These use the environment in `scenarios/requirements.lock.txt`.

- `scenarios/check_scenarios.py`: automatic checks on the drafted scenarios
- `scenarios/select_scenarios.py`: selects the final 50 scenarios per author
- `scenarios/build_anchors.py`: draws the 25 anchors per author and pairs each with two scenarios
- `prompts/build_prompts.py`: builds the 1,000 prompts and the 2,500-call manifest

Re-running the generation needs an OpenRouter key (`OPENROUTER_API_KEY`):

```bash
python generation/runner.py check     # key, credit, pinned endpoints
python generation/runner.py run       # 2,500 calls -> generation/outputs/main
```

Pinned providers and model snapshots may no longer be available, so a new run will not reproduce the deposited texts exactly. The analysis can always be reproduced from the deposited texts.

## Data sources and licence

The literary texts are public-domain works from Project Gutenberg (https://www.gutenberg.org), with the Gutenberg headers and licence text removed. [LICENCE: to be added]

## Citation

[Citation and DOI: to be added on publication]
