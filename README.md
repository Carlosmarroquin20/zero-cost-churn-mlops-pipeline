# Zero-Cost Customer Churn MLOps Pipeline

[![CI](https://github.com/Carlosmarroquin20/zero-cost-churn-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Carlosmarroquin20/zero-cost-churn-mlops-pipeline/actions/workflows/ci.yml)

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
│   └── config.yaml            # Dataset, paths, model, and tracking settings
├── data/                      # DVC-tracked (ignored by Git)
│   ├── raw/
│   └── processed/
├── models/                    # Trained model artifacts (ignored by Git)
├── notebooks/                 # Exploratory analysis
├── reports/                   # Generated evaluation reports (metrics.json)
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI app: health and prediction endpoints
│   │   └── schemas.py         # Request and response schemas
│   ├── config.py              # Configuration loading and path helpers
│   ├── data_ingestion.py      # Download the dataset from Kaggle
│   ├── data_validation.py     # Pandera data contract for the raw data
│   ├── data_preprocessing.py  # Clean and split the dataset
│   ├── features.py            # Feature preprocessing transformer
│   ├── model.py               # Model pipeline and estimator registry
│   ├── evaluation.py          # Classification metrics
│   ├── tracking.py            # MLflow tracking configuration
│   ├── train.py               # Training entry point
│   └── predict.py             # Load the trained model and score customers
├── tests/                     # pytest suite
├── .env.example               # Template for local environment variables
├── .github/workflows/ci.yml   # Lint, tests, and image build on every PR
├── Dockerfile                 # Serving image for the inference API
├── requirements.txt           # Data, training, and tracking dependencies
├── requirements-api.txt       # Inference API runtime dependencies
├── requirements-dev.txt       # All of the above plus test and lint tooling
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

# 4. (Optional) Create a local .env for credentials; it is ignored by Git
cp .env.example .env
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

## Preprocessing

```bash
python -m src.data_preprocessing --config config/config.yaml
```

This validates the raw data against the Pandera contract, cleans it, and writes
stratified `train.csv` and `test.csv` splits to `data/processed/`. Encoding and
scaling happen later, inside the model pipeline, to prevent data leakage.

## Training & experiment tracking

```bash
python -m src.train --config config/config.yaml
```

The training step:

1. Builds a scikit-learn `Pipeline` with the feature preprocessor and the
   classifier set in `config.yaml` (logistic regression by default).
2. Runs stratified k-fold cross-validation on the training split.
3. Fits the model and evaluates it on the test split (ROC AUC, average
   precision, precision, recall, F1, and accuracy).
4. Saves `models/model.joblib` and `reports/metrics.json`.
5. Records the parameters, metrics, and both files in MLflow.

By default, runs are stored locally in `mlflow.db` (SQLite). Browse them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To log runs to the MLflow server included with your DagsHub repository instead,
set these variables in `.env` (or as CI secrets):

```ini
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>
```

## Inference API

The FastAPI service loads `models/model.joblib` at startup and serves churn
predictions. Start it from the repository root:

```bash
uvicorn src.api.main:app --reload
```

Then open <http://localhost:8000/docs> for the interactive documentation.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `200` with the model version, or `503` when no model is loaded |
| `POST` | `/predict` | Score one customer |
| `POST` | `/predict/batch` | Score up to `api.max_batch_size` customers (1,000 by default) |

Request fields use snake_case names and are validated against the same
categories as the raw data contract. For example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "senior_citizen": 0, "partner": "Yes",
       "dependents": "No", "tenure": 1, "phone_service": "No",
       "multiple_lines": "No phone service", "internet_service": "DSL",
       "online_security": "No", "online_backup": "Yes",
       "device_protection": "No", "tech_support": "No",
       "streaming_tv": "No", "streaming_movies": "No",
       "contract": "Month-to-month", "paperless_billing": "Yes",
       "payment_method": "Electronic check",
       "monthly_charges": 29.85, "total_charges": 29.85}'
```

Example response (values are illustrative):

```json
{
  "churn_probability": 0.72,
  "will_churn": true,
  "threshold": 0.5,
  "model_version": "<md5-of-model-file>"
}
```

`model_version` is the MD5 hash of the served model file, the same hash DVC
records for the artifact, so every prediction can be traced back to the exact
model that produced it.

## Container

The image contains only the API runtime: the serving requirements, the source
code, the configuration, and the model directory. Training, tracking, and data
versioning tools stay out of it.

```bash
docker build -t churn-api:local .
```

```bash
docker run --rm -p 8000:8000 churn-api:local
```

The API is then available at <http://localhost:8000/docs>.

Notes:

- `models/` is copied as it is. Build after training to ship a model, or build
  without one: the container still starts and reports itself as unavailable
  until a model is mounted, for example with
  `-v "$(pwd)/models:/app/models:ro"`.
- The listening port comes from the `PORT` environment variable (8000 by
  default), which is what hosting platforms such as Hugging Face Spaces need.
- The container runs as an unprivileged user (UID 1000).
- Serving dependency versions are pinned in `requirements-api.txt` because the
  model is a pickle: it must be loaded with the libraries that produced it.

## Continuous integration

Every pull request, and every push to `master`, runs
[the CI workflow](.github/workflows/ci.yml):

| Job | What it does |
|---|---|
| Lint and tests | black, isort, and flake8 over `src` and `tests`, then the pytest suite on Python 3.10 |
| Serving image | Builds the Docker image, starts the container, and checks that it answers on `/health` |

A new commit on the same branch cancels the previous run, which keeps the free
GitHub Actions minutes for the runs that matter.

## Testing

```bash
pytest
```

Tests run on seeded synthetic data, a temporary local MLflow store, and a
model trained on that synthetic data for the API. MLflow credentials are
removed from the test environment, so no test can reach a remote server.

## License

This project is intended for educational and portfolio purposes.
