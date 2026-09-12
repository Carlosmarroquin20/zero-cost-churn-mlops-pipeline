# Zero-Cost Customer Churn MLOps Pipeline

A production-grade, **100% free-tier** Customer Churn Prediction system, built
end to end with open-source and free tooling: Kaggle, DVC with DagsHub Storage,
MLflow on DagsHub, FastAPI, Docker, GitHub Actions, and Hugging Face Spaces.

## Architecture (target)

```
Kaggle Dataset -> DVC (DagsHub Storage) -> Training Pipeline (MLflow on DagsHub)
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

## Data acquisition & versioning

### 1. Configure Kaggle API credentials

1. Sign in at <https://www.kaggle.com> and open **Settings → API**.
2. Under **Legacy API Credentials**, click **Create Legacy API Key**. This
   downloads a `kaggle.json` file.
3. Place it where the Kaggle client expects it:
   - Windows: `C:\Users\<user>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json` (then `chmod 600` it)

Alternatively, set `KAGGLE_USERNAME` and `KAGGLE_KEY` in a local `.env` file
(see `.env.example`).

### 2. Download the dataset

```bash
python -m src.data_ingestion --config config/config.yaml
```

This stores `data/raw/telco_churn.csv`.

### 3. Track the data with DVC (DagsHub Storage)

Every DagsHub repository includes an S3-compatible storage bucket that DVC can
use as a remote. Create a DagsHub repository and an access token, then run:

```bash
dvc add data/raw/telco_churn.csv
dvc remote add -d dagshub s3://dvc
dvc remote modify dagshub endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify dagshub --local access_key_id <DAGSHUB_TOKEN>
dvc remote modify dagshub --local secret_access_key <DAGSHUB_TOKEN>
dvc push
```

> Always pass `--local` for credentials: it writes them to `.dvc/config.local`,
> which Git ignores. Only the endpoint URL belongs in `.dvc/config`.

## Testing

```bash
pytest
```

## License

This project is intended for educational and portfolio purposes.
