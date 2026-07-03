# Zero-Cost Customer Churn MLOps Pipeline

A production-grade, **100% free-tier** Customer Churn Prediction system, built
end to end with open-source and free tooling: Kaggle, DVC + Google Drive,
MLflow/DagsHub, FastAPI, Docker, GitHub Actions, and Hugging Face Spaces.

## Architecture (target)

```
Kaggle Dataset -> DVC (Google Drive remote) -> Training Pipeline (MLflow/DagsHub)
                                                       |
                                               Model artifact
                                                       |
GitHub Actions (CI/CD) -> Tests + Validation -> Docker -> Hugging Face Spaces
                                                       |
                                             FastAPI inference API
```

## Repository structure

```
zero-cost-churn-mlops-pipeline/
├── config/
│   └── config.yaml          # Dataset, paths, and pipeline parameters
├── data/                    # DVC-tracked (ignored by Git)
│   ├── raw/
│   └── processed/
├── models/                  # DVC-tracked model artifacts
├── notebooks/               # Exploratory analysis
├── src/
│   ├── __init__.py
│   └── data_ingestion.py    # Download the dataset from Kaggle
├── tests/
│   └── test_data_ingestion.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## Local setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash)
# .\.venv\Scripts\Activate.ps1    # Windows (PowerShell)

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Install the pre-commit hooks
pre-commit install
```

## Phase 1 — Data acquisition & DVC

### 1. Configure Kaggle API credentials

1. Sign in at <https://www.kaggle.com> and go to **Settings → API → Create New Token**.
2. This downloads a `kaggle.json` file.
3. Place it where the Kaggle client expects it:
   - Windows: `C:\Users\<user>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

### 2. Download the dataset

```bash
python -m src.data_ingestion --config config/config.yaml
```

This stores `data/raw/telco_churn.csv`.

### 3. Track the data with DVC (Google Drive remote)

```bash
dvc init
dvc add data/raw/telco_churn.csv
dvc remote add -d gdrive_remote gdrive://<GOOGLE_DRIVE_FOLDER_ID>
dvc remote modify gdrive_remote gdrive_use_service_account true
dvc remote modify --local gdrive_remote \
    gdrive_service_account_json_file_path gdrive-service-account.json
dvc push
```

> The service-account JSON file is a secret and is ignored by Git.

## Testing

```bash
pytest
```

## License

This project is intended for educational and portfolio purposes.
