# Online Fraud Transaction Detection — Capstone Starter

This repository implements the Option A capstone: online fraud classification, evaluation, FastAPI, tests, GitHub Actions CI, and MLflow tracking.

## Important
- The assignment's trainer PDF is a separate TinyLlama/QLoRA customer FAQ project. It is not the fraud classifier for this capstone.
- The assignment text says "five models" but explicitly names only Logistic Regression, Decision Tree, and Random Forest. This starter implements those three named models. Ask your trainer whether two additional algorithms are required.
- Use the actual dataset supplied/approved by your trainer. This code expects a CSV with a fraud target column named `isFraud`, `Is_Fraud`, or `is_fraud`. PaySim-style data is supported.
- Do not claim metrics or Azure deployment succeeded until you run the steps and capture your own results.

## 1. Environment
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Add your dataset
Put the CSV at `data/transactions.csv` (create the `data` folder). Do not commit a large/private dataset to GitHub.

## 3. Train and evaluate
```bash
python -m src.train --data data/transactions.csv --target isFraud
```
If your target has another name, use e.g. `--target Is_Fraud`. The script saves `models/fraud_pipeline.joblib`, `reports/metrics.json`, plots, and an evaluation CSV. It logs metrics and artifacts to MLflow locally by default.

To open the local MLflow UI:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then open the URL printed by MLflow (usually http://127.0.0.1:5000).

## 4. Run the API
```bash
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs`. The API schema is generated from the saved `models/fraud_pipeline.joblib` feature columns. Send a JSON object with the same feature names and data types used by the training CSV, excluding the target.

## 5. Test
```bash
pytest -q
```

## 6. GitHub Actions
Create a GitHub repository, push the project, and the workflow at `.github/workflows/ci.yml` runs tests and lint checks on pushes and pull requests.

## 7. Azure ML + MLflow
1. In Azure Portal, create or open an Azure Machine Learning workspace.
2. In Azure ML Studio, open the workspace and find the MLflow tracking URI (workspace MLflow/Tracking URI area; exact UI wording can change).
3. Configure the URI in your terminal before training:
   - macOS/Linux: `export MLFLOW_TRACKING_URI="YOUR_WORKSPACE_MLFLOW_TRACKING_URI"`
   - PowerShell: `$env:MLFLOW_TRACKING_URI="YOUR_WORKSPACE_MLFLOW_TRACKING_URI"`
4. Sign in with the Azure CLI (`az login`) and ensure your identity has permission to use the workspace.
5. Re-run training. Check the Azure ML Studio Jobs/Experiments area for logged runs.
6. Register and deploy only after the local model, API, and tests work. Managed online endpoints can incur charges. Use a small supported instance, check quotas, and delete the endpoint/compute when finished.

This starter's training code logs to whichever MLflow tracking URI is active. Azure endpoint deployment is intentionally a separate step because workspace name, resource group, region, compute quota, and Azure permissions are specific to your account.

## Deliverables checklist
- [ ] Dataset preview, data cleaning, EDA plots
- [ ] Class balance before/after training-only SMOTE
- [ ] Stratified split and cross-validated model comparison
- [ ] Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, PR and ROC curves
- [ ] Best pipeline saved and reloaded; prediction parity checked
- [ ] FastAPI `/predict`, `/health`, Swagger screenshot
- [ ] Pytest results
- [ ] Passing GitHub Actions run
- [ ] MLflow experiment screenshots; Azure workspace/model registration/endpoint screenshots
- [ ] Report with real measured results, limitations, ethics, references
