# RQ2: Generalized Commonsense for GUI Bug Detection

RQ2 compares VisionDroid, VanillaMLLM, and KuiTest with and without generalized
commonsense under Qwen 3.7 Plus and Gemini 3.1 Pro Preview. This produces
`3 baselines x 2 models x 2 conditions = 12` experiment conditions.

The bundled detector JSON files are parsed model outputs, not ground-truth labels.
This module intentionally contains no Precision, Recall, F1, manual assessment,
or inferred labels from directory names.

## Pipeline

1. `sequence_context` extracts situations from the stitched sequence image,
   page descriptions, and action descriptions. VisionDroid and KuiTest share it.
2. `image_only` extracts operations and situations only from the stitched image.
   VanillaMLLM uses it.
3. Retrieval compares each situation with all base and variant situations. A
   generalization record receives `max(base_score, variant_scores)`, followed by
   cosine Top 60, rerank Top 40, and enriched Top 30 storage.
4. The `with_commonsense` detector condition injects Top 10 per situation. The
   detector output schema remains unchanged.

The model used to create the bundled situation files was not reliably
recorded and is therefore marked `not_recorded`. Those files are reused across
both detector models.

KuiTest uses page descriptions in both conditions. Its control is
`bug_with_page_descriptions`; its treatment is `bug_with_commonsense`, changing
only response verification by adding retrieved commonsense.

## Data

The situation files and detector outputs are distributed as compressed
artifacts. From `package/`, restore the RQ2 data and generalized-library
reference files with:

```bash
python extract_artifacts.py --only package_data.zip --only rq2_data.zip
```

- `data/case_manifest.jsonl`: 151 IDs, apps, source datasets, and external asset
  paths; no labels.
- `data/situations/*/extracted`: extracted situation documents.
- `data/situations/*/retrieved`: enriched generalized-commonsense retrieval.
- `data/detector_outputs`: 12 conditions with the same 151 cases each.
- `data/detector_output_manifest.json`: provenance and SHA-256 for all 1,812
  detector JSON files.
- `evaluation/`: intentionally empty for later formal evaluation.

The approximately 821 MB of screenshots, annotations, descriptions, and pickle
test data is not duplicated. `--dataset-root` must point to a directory containing
the `Odin/` and `RegDroid/` datasets named in the case manifest.

## External test datasets

Download the RQ2/RQ4 GUI test datasets from
[Google Drive](https://drive.google.com/drive/folders/1JF9fgd-MAudzztPH-YpYr9kX--PEDFxD)
and extract them to the following recommended location:

```text
package/
`-- data/
    `-- external/
        `-- gui_test_datasets/
            |-- Odin/
            `-- RegDroid/
```

The value passed to `--dataset-root` is the directory that directly contains
`Odin/` and `RegDroid/`. When running from `package/`, use:

```bash
--dataset-root data/external/gui_test_datasets
```

RQ2 and RQ4 can share this extracted directory. Do not place the downloaded
archive itself in `Odin/` or `RegDroid/`; extract its contents first.

## Commands

Run from `package/` after configuring `.env`:

```bash
python rqs/rq2/run.py doctor
python rqs/rq2/run.py situations --dataset-root data/external/gui_test_datasets
python rqs/rq2/run.py retrieve
python rqs/rq2/run.py detect --dataset-root data/external/gui_test_datasets --baseline visiondroid --model qwen --injection both
python rqs/rq2/run.py detect --dataset-root data/external/gui_test_datasets --baseline all --model all --injection both
python rqs/rq2/run.py all --dataset-root data/external/gui_test_datasets
```

Cloud-backed commands support `--ids`, `--limit`, `--workers`, `--overwrite`,
`--dry-run`, and `--top-k`. New outputs and resumable caches are written under
`package/work/rq2/`; bundled experiment results are never modified.
