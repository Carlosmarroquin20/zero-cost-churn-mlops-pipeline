# CLAUDE.md — Project Context & Working Agreement

> This file is the single source of truth for how to work on this project.
> Read it first whenever context is lost. Keep it updated at the end of each phase.

## 1. What we are building

A **production-grade, 100% free-tier Customer Churn Prediction system**, built end
to end as an MLOps portfolio project. The goal is a reproducible pipeline: data
versioning, experiment tracking, a trained model, a REST inference API, tests,
CI/CD, and a public deployment — all at **strictly $0 cost**.

## 2. Non-negotiable constraints

1. **Budget: $0.** Only free tiers. No paid services, ever.
2. **Code quality:** Production-ready Python, PEP 8, modular (Cookiecutter Data
   Science-inspired layout).
3. **Language:** ALL code comments, docstrings, commit messages, and docs in
   **professional English**. (Conversation with the user is in Spanish.)
4. **Testing:** `pytest` unit tests + data validation are part of the architecture,
   written incrementally with each phase — never deferred to the end.

## 3. Tooling (all free tier)

| Concern | Tool | Notes |
|---|---|---|
| Dataset | Kaggle Telco Churn (`blastchar/telco-customer-churn`) | Access via **Kaggle API token** |
| Data/model versioning | DVC + Google Drive remote | Auth via **Service Account** (CI-friendly) |
| Experiment tracking | MLflow on DagsHub | From Phase 3 |
| CI/CD | GitHub Actions | 2,000 min/month free |
| Serving | FastAPI + Docker | — |
| Hosting | **Hugging Face Spaces** (primary) / Render (fallback) | Decide at Phase 7 |

## 4. Key decisions already made

- **Kaggle access:** Kaggle API + `kaggle.json` token (reproducible / CI-friendly).
- **DVC remote auth:** Google Cloud **Service Account** JSON key (works headless in CI).
- **Hosting leaning:** Hugging Face Spaces over Render (more stable free Docker host).

## 5. Master plan & status

| Phase | Objective | Status |
|---|---|---|
| 0 | Scaffolding & tooling (git, venv, linters, pre-commit) | ✅ Done |
| 1 | Data acquisition & DVC + Google Drive | 🟡 Code done; pending user credential steps |
| 2 | Data validation & preprocessing (Pandera/Pydantic + tests) | ⬜ Not started |
| 3 | Training pipeline & MLflow/DagsHub | ⬜ |
| 4 | DVC pipeline (`dvc.yaml` stages) | ⬜ |
| 5 | FastAPI inference service | ⬜ |
| 6 | Containerization (Docker) | ⬜ |
| 7 | CI/CD with GitHub Actions | ⬜ |
| 8 | Deployment & live verification | ⬜ |
| 9 | Hardening & documentation | ⬜ |

### Phase 1 remaining (needs the user's credentials — cannot be automated)
- Place Kaggle token at `C:\Users\carlo\.kaggle\kaggle.json`, then run
  `python -m src.data_ingestion --config config/config.yaml`.
- Create GCP Service Account + Drive folder, then `dvc add` / `dvc remote add` /
  `dvc push`. Verify with the delete → `dvc pull` → `dvc status` round-trip.

## 6. Repository layout

```
config/config.yaml        # Dataset, paths, pipeline parameters (no hard-coded paths in code)
data/raw, data/processed  # DVC-tracked (git-ignored)
models/                   # DVC-tracked artifacts
src/                      # Python modules (data_ingestion.py, ...)
tests/                    # pytest suite
.pre-commit-config.yaml   # black, isort, flake8, misc hooks
pyproject.toml            # black/isort/pytest config; .flake8 for flake8
```

## 7. How to run things (Windows + Git Bash)

The virtual environment lives at `.venv/`. Use its interpreter explicitly:

```bash
.venv/Scripts/python.exe -m pytest          # run tests
.venv/Scripts/python.exe -m black src tests # format
.venv/Scripts/python.exe -m isort src tests
.venv/Scripts/python.exe -m flake8 src tests
.venv/Scripts/dvc.exe status                # DVC
```

Or activate: `source .venv/Scripts/activate`.

## 8. Conventions & workflow

- **Incremental:** one phase at a time. Do NOT jump ahead. Wait for the user to
  say "vamos con la fase N" before starting it.
- **Commits:** only when the user explicitly asks. Use Conventional Commits in
  English (`feat:`, `fix:`, `ci:`, `docs:`, `test:`).
  End commit messages with the required `Co-Authored-By` trailer.
- **Secrets:** NEVER commit credentials. Secrets are git-ignored (see `.gitignore`);
  use `.env` (from `.env.example`) and DVC `--local` config. DagsHub/MLflow tokens
  go in `.env` or CI secrets, never in code.
- **DVC vs Git:** Git tracks code + `.dvc` pointer files; DVC tracks data/models.
- **Config-driven:** read paths/params from `config/config.yaml`, never hard-code.
- **Update this file** at the end of each phase (status table + new decisions).
```
