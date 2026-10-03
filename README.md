# AutoTestSense Reproduction Package

AutoTestSense mines reusable testing commonsense from mobile-app bug reports
and supplies that knowledge to GUI bug detectors. This package contains the
code, prompts, fixed inputs, reference artifacts, and experiment indexes needed
to reproduce the workflow described in:

> *Scaling Commonsense Test Oracles: Automatically Mining Testing Commonsense
> from Mobile App Bug Reports*

The released reference library contains **1,399 generalized commonsense rules**
covering **3,708 instance-level commonsense items** extracted from **4,634 bug
reports** across **868 Android applications**.

## What AutoTestSense Does

The paper presents four conceptual phases:

1. **Instance-level commonsense extraction** identifies reusable behavioral
   expectations in individual bug reports.
2. **Commonsense generalization** groups compatible instances and derives a
   shared rule while preserving meaningful variants.
3. **Retrieval-guided consolidation** assigns initially ungrouped instances to
   an existing rule or creates a new rule.
4. **Commonsense-guided bug detection** retrieves rules applicable to a test
   situation and injects them into a GUI bug detector.

For reproducibility, the implementation exposes these phases as seven smaller
stages:

| Stage | Command | Purpose |
|---|---|---|
| 1. **Instance-Level Commonsense Extraction** | `stage1` | Fetch the fixed issue set and extract candidate commonsense. |
| 2. **Similarity and Community Discovery** | `stage2` | Embed instances, retrieve neighbors, build an SNN graph, and refine communities. |
| 3. **Initial Commonsense Generalization** | `stage3` | Generalize each local community with an LLM. |
| 4. **Global Commonsense Generalization** | `stage4` | Find and reconcile related rules across local communities. |
| 5. **Commonsense Consolidation** | `stage5` | Serially place every unassigned instance into the evolving library. |
| 6. **Commonsense Library Assembly** | `stage6` | Produce the final records and situation-retrieval index. |
| 7. **Commonsense Retrieval** | `stage7` | Retrieve from the generalized library or the instance-level ablation index. |

## What Is Included

The package is self-contained with respect to source code and prompts. It does
not import from the repository's `src/`, `script/`, or `prompt/` directories.

| Artifact | Location | Contents |
|---|---|---|
| Package data | [`artifacts/package_data.zip`](artifacts/package_data.zip) | Issue manifest, 3,708 extracted instances, 1,399-rule library, and embeddings. |
| RQ2 data | [`artifacts/rq2_data.zip`](artifacts/rq2_data.zip) | Case manifest, situations, and detector outputs. |
| RQ4 data | [`artifacts/rq4_data.zip`](artifacts/rq4_data.zip) | Situations and Gemini ablation outputs. |

GitHub report bodies and the external GUI test datasets are not duplicated in
this package. Stage 1 fetches current issue content from the URLs in the
manifest. The GUI test datasets used by RQ2 and RQ4 can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1JF9fgd-MAudzztPH-YpYr9kX--PEDFxD).
Download and extract them as described in the RQ2 and RQ4 READMEs.

Packaged data is stored in three ZIP archives under
[`artifacts/`](artifacts/).

## Restore Packaged Data

After cloning the repository, run the following command from `package/` before
`doctor` or reproduction commands:

```bash
python extract_artifacts.py
```

The script verifies every archive against
[`artifacts/SHA256SUMS.txt`](artifacts/SHA256SUMS.txt) and restores files to
their expected `data/` and `rqs/` locations. Existing files are skipped unless
`--force` is supplied.

To verify downloads without extracting them:

```bash
python extract_artifacts.py --verify-only
```

To restore only selected artifacts:

```bash
python extract_artifacts.py --only package_data.zip
python extract_artifacts.py --only rq2_data.zip
```

The archive contents and restored paths are listed in
[`artifacts/README.md`](artifacts/README.md). Manual extraction is also valid:
all archive entries are relative to `package/`, so extract each ZIP directly
into the `package/` directory.

## Installation

Python **3.11** is the supported runtime.

```bash
cd package
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Create the environment file:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux or macOS
cp .env.example .env
```

Fill only the credentials needed by the stages you intend to run, then verify
the package:

```bash
python reproduce.py doctor
```

`doctor` checks Python, dependencies, prompts, reference counts, member
coverage, and vector dimensions. It does not call a remote API.

## Choose a Reproduction Path

### 1. Inspect or validate the released artifacts

No cloud call is required:

```bash
python reproduce.py doctor
```

Start with the final library file linked in the artifact table above.

### 2. Rebuild from the extracted commonsense checkpoint

The bundled 3,708-item checkpoint lets you skip issue fetching and extraction:

```bash
python reproduce.py all --start-stage 2 --end-stage 6
```

This path reruns embedding, graph construction, local and global
generalization, serial consolidation, and final library assembly. It requires
the embedding, reranking, and LLM services configured in `.env`.

For a small request-level check before a full run:

```bash
python reproduce.py all --start-stage 2 --end-stage 6 --limit 5
```

### 3. Rebuild from the fixed issue manifest

```bash
python reproduce.py all --start-stage 1 --end-stage 6
```

This path fetches GitHub issue title/body content before running the full
library-construction pipeline. A GitHub token and extraction-model credentials
are required. Issue edits, deletions, and permission changes can make newly
fetched text differ from the historical input.

### 4. Run commonsense retrieval only

Stage 7 automatically uses the packaged final library and embeddings when no
new Stage 6 output exists:

```bash
# Generalized library
python reproduce.py stage7 --dataset visiondroid --mode library

# Direct retrieval from all 3,708 extracted instances
python reproduce.py stage7 --dataset visiondroid --mode instance_level

# Run both representations
python reproduce.py stage7 --dataset vanilla_mllm --mode both
```

Retrieval results are written under `work/stage7/` and include the enriched Top
30 candidates for each input situation.

### 5. Reproduce the paper experiments

The research-question modules are independent entry points:

| Module | Experiment | Entry point |
|---|---|---|
| [RQ1](rqs/rq1/) | Quality sample: 302 generalized rules and their 848 source instances. | Metadata and evaluation index |
| [RQ2](rqs/rq2/) | Three GUI bug detectors, two MLLMs, with and without generalized commonsense. | `python rqs/rq2/run.py doctor` |
| [RQ4](rqs/rq4/) | Gemini ablation using the 3,708 instance-level items instead of generalized rules. | `python rqs/rq4/run.py doctor` |

RQ2 and RQ4 share the external GUI test datasets available from
[Google Drive](https://drive.google.com/drive/folders/1JF9fgd-MAudzztPH-YpYr9kX--PEDFxD).
Their READMEs define the extraction location, required case assets, and commands.
Bundled detector outputs are model outputs, not ground-truth labels or
precomputed evaluation metrics.

## Pipeline Details

### Stage 1: extraction

Each fixed issue is fetched once per unique URL. The extraction prompt produces
structured candidates, and the pipeline retains records marked as candidate
commonsense violations.

### Stage 2: community discovery

The pipeline embeds `situation` and `violated_commonsense_rule`, performs cosine
Top-60 recall and Top-30 reranking, and builds a non-mutual shared-nearest-neighbor
graph with a minimum of 15 shared neighbors. Louvain community discovery uses
resolution 0.8. Leiden recursively divides large communities until each LLM
batch contains at most 25 instances.

### Stage 3: local generalization

An LLM examines each refined community and emits generalized rules, equivalent
members, and variant groups with their differences from the shared rule.

### Stage 4: global generalization

Rules from separate communities are compared using weighted situation/rule
embeddings (0.35/0.65), Top-20 recall, and Top-10 reranking. Mutual Top-3 links
with scores of at least 0.70 form candidates for a second LLM generalization.

### Stage 5: serial consolidation

Unassigned instances are processed one at a time against the current library.
Each decision uses cosine Top-20 recall, Top-10 reranking, a 0.70 candidate
threshold, and at least three candidates. New or revised base rules are embedded
before processing the next instance.

### Stage 6: library assembly

The pipeline exports the final records and builds the situation index. Base-rule
situations receive new embeddings; variant situations reuse their original
instance embeddings.

### Stage 7: retrieval

For the generalized-library method, a record's recall score is the maximum over
its base and variant representations. The highest-scoring representation is
sent to reranking. The ablation retrieves directly from all instance-level
items. Both modes use cosine Top 60, rerank Top 40, and save an enriched Top 30.

## Command Reference

```bash
python reproduce.py stage1
python reproduce.py stage2
python reproduce.py stage3
python reproduce.py stage4
python reproduce.py stage5
python reproduce.py stage6
python reproduce.py stage7 --dataset visiondroid --mode both
python reproduce.py all --start-stage 1 --end-stage 7 --dataset visiondroid --mode both
```

Common options:

- `--limit N`: process a bounded subset for validation.
- `--workers N`: set concurrency where the stage supports it.
- `--dry-run`: resolve and print stage commands without calling APIs.
- `--overwrite`: replace completed outputs instead of resuming.

All generated artifacts are written to `work/`. Each executed stage writes a
manifest under `work/run_manifests/` containing arguments, model identifiers,
input hashes, elapsed time, output summaries, and API usage. Files under
`data/checkpoints/` and `data/reference/` are never modified.

## Repository Map

```text
package/
|-- reproduce.py                         unified seven-stage entry point
|-- extract_artifacts.py                 packaged-data extraction
|-- artifacts/                           packaged data archives
|-- prompts/                             versioned prompts and prompt index
|-- commonsense_repro/
|   |-- extraction/                      Stage 1
|   |-- clustering/                      Stage 2
|   |-- commonsense_generalization/      Stages 3 and 4
|   |-- commonsense_consolidation/       Stage 5
|   |-- library_construction/            Stage 6
|   |-- retrieval/                       Stage 7
|   `-- common/                          shared I/O and API utilities
|-- data/                                fixed inputs, checkpoints, references
|-- rqs/                                 paper experiment artifacts
`-- work/                                generated outputs, logs, and manifests
```

Prompt roles, caller modules, placeholders, and output contracts are documented
in [`prompts/README.md`](prompts/README.md).

## Reproducibility Notes

- LLM generations are nondeterministic. The package reproduces the data flow,
  prompts, parameters, interfaces, and validation invariants, not byte-identical
  generated text.
- A fresh Stage 1 run depends on the current state of the linked GitHub issues.
- Network-backed stages are resumable by default. Use `--overwrite` only when a
  clean rerun is intended.
- Published reference artifacts remain read-only; new runs are isolated under
  `work/`.
