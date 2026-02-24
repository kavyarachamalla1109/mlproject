# agribot-mlops-mini

A complete, beginner-friendly MLOps mini-project for crop recommendation classification using a tabular agriculture dataset, scikit-learn, and a lightweight FastAPI inference app.

## 1) Project overview

This repository demonstrates a practical, end-to-end ML workflow:

- Validate config and data
- Preprocess tabular data
- Train a baseline `RandomForestClassifier`
- Tune hyperparameters (`GridSearchCV` or `RandomizedSearchCV`)
- Evaluate model quality
- Save production-friendly artifacts for inference
- Package a downloadable inference ZIP bundle
- Run all checks in one GitHub Actions workflow

It is intentionally simple and local-first (CPU-only), while keeping clean modular design for future scaling.

## 2) Use case: AgriBot crop recommendation

Given soil and weather features (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`), the model predicts a crop label (e.g., rice, maize, chickpea) for basic decision support.

## 3) Folder structure

```text
agribot-mlops-mini/
  main.py                          # FastAPI app entrypoint (python main.py)
  templates/
    index.html                     # Simple HTML UI
  README.md
  requirements.txt
  .gitignore
  .env.example
  Makefile
  configs/
    train_config.yaml
  data/
    raw/
      crop_recommendation_sample.csv
    processed/
      .gitkeep
  artifacts/
    .gitkeep
  src/
    __init__.py
    main.py                        # training orchestrator
    config.py
    validate.py
    preprocess.py
    train.py
    tune.py
    evaluate.py
    predict.py
    artifacts.py
    deploy.py
    utils.py
  tests/
    test_config.py
    test_validate.py
    test_train_smoke.py
  .github/
    workflows/
      agribot-pipeline.yml
```

## 4) Setup instructions (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use Makefile:

```bash
make install
```

## 5) How to run training pipeline

```bash
python -m src.main --config configs/train_config.yaml
```

Equivalent Make target:

```bash
make train
```

## 6) What you get after training

`artifacts/` will contain:

- `agribot_model.pkl` (primary model artifact)
- `data_validation_report.json`
- `metrics.json`
- `metrics.md`
- `predictions_sample.csv`
- `best_params.json`
- `run_summary.json`
- `agribot_inference_bundle.zip` (downloadable app bundle)

## 7) Run local prediction API + HTML UI with simple command

After training, start the app with:

```bash
python main.py
```

Then open:

- Web UI: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

This app loads `artifacts/agribot_model.pkl` and returns crop predictions.

## 8) Predict from CLI using pickle file

```bash
python -m src.predict \
  --model artifacts/agribot_model.pkl \
  --input data/raw/crop_recommendation_sample.csv \
  --output artifacts/predictions_output.csv
```

The output CSV contains original features plus a `prediction` column.

## 9) Download and run ZIP bundle

After CI pipeline run, download artifacts and extract `agribot_inference_bundle.zip`.

Inside extracted folder:

```bash
python -m pip install -r requirements.txt
python main.py
```

The bundle includes model, HTML template, requirements, and quickstart instructions.

## 10) GitHub Actions workflow behavior

Workflow file: `.github/workflows/agribot-pipeline.yml`

Triggers:

- `push`
- `pull_request`
- `workflow_dispatch`

Steps:

1. Setup Python 3.11
2. Install dependencies
3. Run `pytest`
4. Run full pipeline (`python -m src.main`)
5. Build GitHub Step Summary from generated artifacts
6. Upload `artifacts/*` (including zip bundle)

Step summary includes:

- Dataset path and shape
- Validation status
- Model type
- Tuning details and best params
- Accuracy/Precision/Recall/F1
- Artifact names
- Local run command

## 11) Customize dataset/config

Edit `configs/train_config.yaml` to change:

- `data_path`
- split and seed (`test_size`, `random_state`)
- tuning strategy and search space
- `output_dir`
- metric averaging strategy

Dataset requirements:

- CSV format
- Must include `N,P,K,temperature,humidity,ph,rainfall,label`
- Target column should match `target_column`

## 12) Future roadmap (toward a larger MLOps control tower)

- Introduce dataset versioning and schema contracts
- Add richer drift and quality checks
- Expand model choices and compare champion/challenger models
- Add packaging for model serving
- Add scheduled retraining
- Integrate experiment tracking and model registry
- Extend CI with linting/type checks and release automation

## Helpful commands

```bash
make test
make train
make clean
python main.py
```
