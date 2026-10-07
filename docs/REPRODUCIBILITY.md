# Reproducibility

The current main study uses Arena Human Preference 140k, pinned to revision `6322995ab34d7c2693e3f47dd13fa5caa0789a74`. Seven source files and SHA-256 hashes are recorded in `arena_data_provenance.json`; cleaned split text hashes are pinned in `arena_processed_hashes.json`. Source content stays in ignored local directories.

## Fresh-clone reproduction

Use Python 3.12 and install `requirements-lock.txt` into a virtual environment. Run these commands from the repository root:

```sh
python src/arena_data.py download
python src/arena_data.py prepare
python src/arena_train.py train
python src/arena_train.py domain
python src/arena_train.py evaluate
python src/arena_analyze.py
python -m unittest discover -s tests -v
python src/build_report.py
```

The published non-text split manifest is included for auditing. In a fresh clone, `prepare` regenerates the ignored split text and requires the new manifest to match the published assignments exactly. It refuses to overwrite existing local split text; use a separate clone for full reproduction.

Training caches are accepted only when the checkpoint, validation probabilities, run metadata, configuration, and split hash match. New numerical outcomes can differ across devices or library versions. If rerunning model selection chooses another candidate, the selection assertion stops instead of silently changing the published experiment; preserve the original freeze and perform the rerun in a separate experimental copy.

`domain` must run before `evaluate`: it trains the non-code holdout and class-count-matched controls, then records all checkpoint hashes. `evaluate` verifies those hashes before loading test text. Re-running test evaluation is only for replay verification, not configuration selection.

`arena_analyze.py` independently checks group disjointness, validation selection, 31 test prediction files, and two CPU checkpoint replays. The test-set cluster bootstrap holds trained models fixed; its intervals are not total uncertainty over model selection and future distributions.

## Local prediction

```sh
python src/predict.py --response 'Your example answer goes here.'
python src/predict.py --checkpoint models/arena_lstm_output_s1448.pt --response 'Your example answer goes here.'
```

The printed values are uncalibrated, closed-set scores for four specific historical releases, not reliable attribution probabilities for arbitrary text.

## Historical UltraFeedback study

`src/download_data.py`, `src/prepare_data.py`, `src/run_experiments.py`, `src/iterate.py`, and corresponding results remain intact. `src/build_legacy_report.py` rebuilds the archived optimized report. The two datasets and label spaces are not pooled. Default three-class behavior in `models_v2.py` remains compatible with existing weights; the Arena models explicitly request four classes.

## Artifacts

The report is generated with ReportLab. Poppler is used for visual inspection and is not required to generate the PDF. Source course PDFs, raw conversations, local environments, and full training checkpoints are excluded from publication. Demonstration weights, non-text predictions, metrics, source hashes, and manifests are included.
