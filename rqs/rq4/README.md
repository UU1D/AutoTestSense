# RQ4: Instance-Level Commonsense Ablation

RQ4 measures the effect of removing commonsense generalization from the bug
detection pipeline. Instead of retrieving generalized commonsense records, it
retrieves directly from all 3,708 instance-level commonsense records.

The bundled experiment contains 151 cases for each of VisionDroid,
VanillaMLLM, and KuiTest, all executed with Gemini 3.1 Pro Preview. The matching
generalized-commonsense detector outputs are stored under RQ2 and are not
duplicated here. No ground-truth labels or aggregate metrics are included.

The situation files and detector outputs are distributed as compressed
artifacts. From `package/`, restore the RQ4 data and instance-level vectors with:

```bash
python extract_artifacts.py --only package_data.zip --only rq4_data.zip
```

## Retrieval and injection

For every extracted test situation, the pipeline computes cosine similarity
against the 3,708 situation embeddings, recalls Top 60, reranks Top 40, and
stores the complete Top 30. The detector receives the Top 10 records, including
each record's applicable situation and expected behavior.

- VisionDroid and KuiTest use the `sequence_context` situation representation.
- VanillaMLLM uses the `image_only` situation representation.
- KuiTest retains its page-description flow. Flat records are adapted as
  base-only singleton entries solely to reuse its page-to-action renderer; no
  generalized family or variant relation is inferred.

## Commands

Download the GUI test datasets from
[Google Drive](https://drive.google.com/drive/folders/1JF9fgd-MAudzztPH-YpYr9kX--PEDFxD)
and extract them to `package/data/external/gui_test_datasets/`. The resulting
directory must directly contain `Odin/` and `RegDroid/`. RQ4 uses the same
external test data as RQ2, so only one extracted copy is needed.

Run these commands from `package/`:

```bash
python rqs/rq4/run.py doctor
python rqs/rq4/run.py retrieve --limit 5 --dry-run
python rqs/rq4/run.py detect --baseline visiondroid --dataset-root data/external/gui_test_datasets --limit 5 --dry-run
python rqs/rq4/run.py detect --baseline all --dataset-root data/external/gui_test_datasets
python rqs/rq4/run.py all --dataset-root data/external/gui_test_datasets
```

Common options are `--ids`, `--limit`, `--workers`, `--top-k`, `--overwrite`,
and `--dry-run`. New outputs are isolated under
`work/rq4/gemini-3.1-pro-preview/`.

Gemini execution uses `RQ2_GEMINI_API_KEY`, `RQ2_GEMINI_URL`, and
`RQ2_GEMINI_MODEL` from `package/.env`. The API model identifier used is
`gemini-3.1-pro-preview-medium`; the paper-facing name is
`gemini-3.1-pro-preview`.

## Bundled artifacts

- `data/situations/sequence_context/retrieved/`: 151 enriched retrieval files.
- `data/situations/image_only/retrieved/`: 151 enriched retrieval files.
- `data/detector_outputs/gemini-3.1-pro-preview/`: 453 parsed detector outputs.
- `data/detector_output_manifest.json`: source mapping and SHA-256 hashes.

The source dataset images and pickle files are not bundled. Supply the extracted
dataset directory through `--dataset-root` when rerunning the detectors.
