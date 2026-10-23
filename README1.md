# Multimodal Explainable Deep Learning Framework for Early Depression Detection Using Speech, Facial Expression, and Text Analysis

Research prototype for exploring participant-level depression screening from interview speech, video-derived facial-region descriptors, and transcript text. It is not a diagnostic or clinical tool. Use only data you are authorized to process, and do not use model scores to make decisions about people.

## Project status while DAIC-WOZ access is pending

The repository contains a runnable pipeline, but it does **not** contain DAIC-WOZ or participant data. Do not describe generated/demo results as DAIC-WOZ results. The neural model is a compact multimodal MLP over fixed MFCC/spectral audio summaries, sampled HOG video summaries, and TF-IDF transcript vectors. It is a deep-learning fusion prototype, not an end-to-end raw audio/video transformer. The video descriptors are not emotion labels or validated depression markers. The explanation output is modality ablation sensitivity, not causal evidence.

## What to edit when you receive the authorized data

| File / location | What you update |
|---|---|
| `data/raw/` | Place permitted recordings, videos, and transcripts here. Keep this private; it is ignored by Git. |
| `data/manifest.csv` | Add one row per participant with the exact participant ID, the matching audio/video/transcript paths, and documented binary label. Never infer labels from filenames. |
| `configs.yaml` | Record the chosen seed, split proportions, label definition, release/version, and preprocessing choices for your study. |
| `src/features.py` | Change feature extraction only if your authorized file formats, sampling policy, or preprocessing differ. Document each change. |
| `src/train_deep.py` | Change neural branch/fusion architecture or training settings. Keep all participants in exactly one split. |
| `app/app.py` | Change project title/text or local results display. Do not use it as a clinical prediction interface. |
| `README.md` / report | Replace pending-access statements with the actual approved release and protocol, and record limitations and results. |

DAIC-WOZ access is controlled; request it from the [official DCAPS/DAIC-WOZ site](https://dcapswoz.ict.usc.edu/) and follow its license and data handling requirements. File naming and available modalities can differ by release. Inspect the release documentation first, then map files manually into the manifest. Do not upload restricted data to an unapproved service.

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12 (64-bit) for the broadest package compatibility:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked in the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

PyTorch installation can vary by CPU/GPU and Python version. If `pip install -r requirements.txt` cannot resolve `torch`, install the matching build from the [official PyTorch selector](https://pytorch.org/get-started/locally/) and then install the other requirements.

## Data manifest

### Check the software before dataset approval

After installing dependencies, you can create synthetic files and run the same pipeline to check that extraction, training, and output writing work:

```powershell
python -m src.create_demo_data --participants 30
python -m src.make_splits --manifest data/demo_manifest.csv --out-dir data/demo_splits --seed 42
python -m src.train_deep --manifest data/demo_manifest.csv --splits-dir data/demo_splits --out-dir artifacts/synthetic_smoke_test --epochs 3 --data-source synthetic
```

The signals are artificial and class-correlated solely to exercise the code. The resulting metrics have no research meaning. Never cite them in a report, poster, thesis, or presentation.

Create `data/manifest.csv` using this exact schema:

```csv
participant_id,audio_path,video_path,transcript_path,label
```

Example format only (fictional paths and IDs; not data):

```csv
participant_id,audio_path,video_path,transcript_path,label
P001,data/raw/P001.wav,data/raw/P001.mp4,data/raw/P001.txt,0
P002,data/raw/P002.wav,data/raw/P002.mp4,data/raw/P002.txt,1
```

Paths may be absolute or relative to the project root. This starter requires all three modalities for every participant; it currently rejects incomplete modality rows rather than imputing a missing modality. There must be one aggregate row per participant. Record in your report exactly what `0` and `1` mean, the label source/threshold, and how files were matched.

Validate the manifest and paths:

```powershell
python -m src.validate_manifest --manifest data/manifest.csv
```

Validation checks existence and schema; it cannot verify that a transcript, recording, and label belong to the same person. Check that mapping against the authorized release materials.

## Participant-level split and training

```powershell
python -m src.make_splits --manifest data/manifest.csv --out-dir data/splits --seed 42
python -m src.train_deep --manifest data/manifest.csv --splits-dir data/splits --out-dir artifacts/deep_run_001 --epochs 40 --seed 42 --data-source authorized
```

The split is stratified at participant level (70/15/15). Training uses AdamW and binary cross entropy with class weighting, and early stopping on validation loss. The 0.5 threshold is fixed in this starter; do not tune on test data. Outputs:

- `artifacts/deep_run_001/model.pt`: trained network weights and input dimensions.
- `artifacts/deep_run_001/preprocessing.joblib`: fitted feature scalers/imputers and text vectorizer.
- `artifacts/deep_run_001/metrics.json`: validation/test metrics and modality ablation sensitivity.

The existing `python -m src.train` command is retained as a simpler logistic-regression baseline. Report it separately from the MLP. Both paths require complete modalities and sufficient examples in each class. Small pilot runs are for pipeline debugging, not final evaluation. For an academic result, consider repeated participant-stratified cross-validation, confidence intervals, calibration, external validation, and a predeclared missing-data policy.

## Explainability and limitations

The current explanation is a modality ablation: it sets one standardized modality input to its neutral value and measures the change in held-out F1. This gives a coarse view of model reliance for this test set, not an explanation of an individual decision, proof of causality, or evidence of a biological/clinical mechanism. Do not present it as SHAP or feature-level attribution. Add a validated attribution method only after documenting its assumptions and validating it for the chosen model and data.

The dataset may be small, imbalanced, and demographically limited. Audio, expression, language, recording conditions, disability, and interview context may introduce confounding. Any report should state release/version, inclusion criteria, label protocol, participant counts, split seed, modality availability, preprocessing, all metrics, uncertainty, and limitations. Seek supervisor/ethics guidance for handling sensitive data.

## App

```powershell
streamlit run app/app.py
```

The app displays metrics from a local experiment file, if present. Verify its `data_source` field before interpreting any result. It does not diagnose or produce person-level predictions.
