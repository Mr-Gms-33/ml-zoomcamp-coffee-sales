# Coffee Vending Sales — ML Zoomcamp Midterm Project

**Course:** [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)  
**Author:** Mounir

End-to-end machine learning project: explore POS transaction logs from a coffee vending machine, train and tune several tabular models, export training to a script, and serve predictions through a containerized web API.

---

## Problem statement

Small retail operators (vending machines, kiosk coffee points) need to **stock the right drinks at the right times** without guessing from raw ticket logs. This project uses real transaction data to learn patterns in **when** people buy **which** drink and **how** they pay.

### Business use case

A deployed model can support:

- **Inventory planning** — prioritize restocking SKUs that are likely to sell in the next time window (e.g. morning vs afternoon).
- **Promotion design** — focus card vs cash incentives when payment mix shifts.
- **Operational dashboards** — attach predicted drink mix to hourly staffing or machine refill routes.

### Machine learning task

**Multiclass classification:** given a transaction context, predict **`coffee_name`** (drink type).

| Role | Columns (examples) |
|------|----------------------|
| **Features** | Datetime-derived fields (`hour`, `day_of_week`, `month`, …), `cash_type` (`card` / `cash`), and optionally anonymized `card` id for repeat-buyer signal |
| **Target** | `coffee_name` |

> **Note:** `money` is largely determined by product and payment type; it is better suited to regression or EDA than as the sole classification target. The notebook documents this choice.

Models are compared with **accuracy** and **macro-F1** (important when drink classes are imbalanced). The best tuned pipeline is saved and exposed via `POST /predict`.

**Modeling note:** Only **machine 1** (`index_1.csv`) is used for training because machine 2 sells a different product catalog with sparse overlap. After tuning, hold-out **macro-F1 ≈ 0.23** — drink choice is weakly determined by time/payment alone (expected for this feature set). See `models/metrics.json` after running `train.py`.

---

## Dataset

Source: [Coffee Sales (Kaggle)](https://www.kaggle.com/datasets/ihelon/coffee-sales) by ihelon — open POS logs from a vending machine (March 2024 onward, updated over time).

| File | Description |
|------|-------------|
| `archive/index_1.csv` | Machine 1 — `date`, `datetime`, `cash_type`, `card`, `money`, `coffee_name` |
| `archive/index_2.csv` | Machine 2 — same schema (no `card` column in some rows) |

**Included in this repo:** `archive/` (snapshot for reproducibility).

**Refresh from Kaggle** (optional):

```bash
pip install kaggle
# Configure ~/.kaggle/kaggle.json (API token from Kaggle account settings)
kaggle datasets download -d ihelon/coffee-sales -p data --unzip
```

---

## Repository layout (deliverables)

| Item | Path | Status |
|------|------|--------|
| Problem description & runbook | `README.md` | Done |
| Data | `archive/` | Committed |
| EDA + model selection | `notebook.ipynb` | Done |
| Training script | `train.py` | Done → `models/pipeline.joblib` |
| Prediction API | `predict.py` | Done (Flask, port **9696**) |
| Dependencies | `requirements.txt` | Done |
| Container | `Dockerfile` | Done |
| Local API test | `predict-test.py` | Done |
| Shared feature logic | `data_utils.py` | Used by `train.py` / `predict.py` |

Notebook checklist (for full rubric points):

1. Load & merge machines; parse `datetime`; handle missing `card`.
2. EDA — class balance, sales by hour/day, payment mix, price ranges, missing values.
3. Feature importance — mutual information / correlation with encoded target; tree importances.
4. Train **≥2 families** (e.g. logistic regression + random forest) with **hyperparameter tuning** (e.g. `RandomizedSearchCV`).
5. Select final model; document metrics on hold-out set.

---

## Environment setup

Python **3.10+** recommended (tested targets: 3.11).

```bash
cd "C:\Users\mouni\Desktop\Data Analyst\DTC\ML_Zoomcamp\Proj"

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Planned dependencies** (pin versions in `requirements.txt` when created):

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- API: `flask` or `fastapi` + `uvicorn`
- Persistence: `joblib` (or `pickle`)

---

## How to run

### 1. Exploratory analysis and model experiments

```bash
jupyter notebook notebook.ipynb
```

### 2. Train final model (after `train.py` exists)

Trains on the full training strategy defined in the notebook and writes an artifact (e.g. `models/pipeline.joblib`).

```bash
python train.py
```

### 3. Run API locally

```bash
python predict.py
```

Default service URL: `http://localhost:9696` (adjust in script if needed).

**Example request** (Flask-style JSON body):

```bash
# PowerShell
Invoke-RestMethod -Method Post -Uri http://localhost:9696/predict -ContentType "application/json" -Body '{"hour":10,"day_of_week":0,"month":3,"cash_type":"card"}'

# curl (cmd / Git Bash)
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"hour\": 10, \"day_of_week\": 0, \"month\": 3, \"cash_type\": \"card\"}"
```

```bash
python predict-test.py
```

**Expected response shape:**

```json
{
  "coffee_name": "Latte",
  "probabilities": {
    "Latte": 0.42,
    "Americano": 0.18
  }
}
```

Optional: `python predict-test.py` with a fixed sample payload.

### 4. Docker

```bash
docker build -t coffee-sales-api .
docker run --rm -p 9696:9696 coffee-sales-api
```

Document the exact port and entrypoint in `Dockerfile` when added.

---

## Deployment (bonus / rubric)

| Level | What reviewers expect |
|-------|------------------------|
| Local Docker | `Dockerfile` + README steps above |
| Cloud (optional) | Live URL **or** screenshot/video of `curl` / Swagger UI |

Examples peers used: Google Cloud Run, Hugging Face Spaces (Gradio), Render, Streamlit Cloud.

- **GitHub (source):** https://github.com/Mr-Gms-33/ml-zoomcamp-coffee-sales  
- **Gradio UI (Hugging Face Space):** https://huggingface.co/spaces/Mr-Gms-33/coffee-sales-midterm — create or sync via steps below  
- **Flask API (Render):** [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Mr-Gms-33/ml-zoomcamp-coffee-sales) → after deploy, URL like `https://coffee-sales-api.onrender.com`

### Hugging Face Space (Gradio — recommended for reviewers)

1. Sign in at [huggingface.co](https://huggingface.co) and run `pip install huggingface_hub` then `hf auth login`.
2. From this folder: `python deploy_hf_space.py` (uploads `app.py`, model, and data).
3. Or manually: **Create Space** → SDK **Gradio** → link GitHub repo → set **App file** to `app.py`.

Space README metadata template: `README_HF.md` (copy to Space `README.md` if needed).

### Render (Flask API in Docker)

Repo includes `render.yaml`. Use the Deploy button above or connect the GitHub repo in the Render dashboard (Web Service → Docker).

**Local API test against cloud:** `python predict-test.py https://YOUR-SERVICE.onrender.com/predict`

---

## Evaluation alignment (max 16 points)

This project is structured to match the [official criteria](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/projects/README.md):

| Criterion | How this repo addresses it |
|-----------|----------------------------|
| Problem description | Context, task, and API use case above |
| EDA | `notebook.ipynb` — target balance, temporal patterns, feature analysis |
| Model training | Multiple models + tuning documented in notebook; `train.py` for final fit |
| Notebook → script | `train.py` mirrors notebook final pipeline |
| Reproducibility | Data in `archive/` + pinned `requirements.txt` |
| Deployment | `predict.py` web service |
| Dependencies | `requirements.txt` + venv instructions |
| Containerization | `Dockerfile` + build/run commands |
| Cloud | Optional URL or media in this README |

**Forbidden course datasets (not used here):** car prices, telco churn, credit risk, clothing images.

---

## Peer review

You must **review 3 peer projects** to pass. When reviewing others, try their README steps; use [nbviewer](https://nbviewer.org/) if GitHub does not render notebooks.

---

## References

- Course projects guide: https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects  
- Dataset: https://www.kaggle.com/datasets/ihelon/coffee-sales  
- Example midterm READMEs: [Manufacturing Uptime (FastAPI)](https://github.com/mathurshubham/Manufacturing-Uptime-Service), [Heart disease (Flask)](https://github.com/nilarte/ml-zoomcamp-midterm-project-2024)

---

## License & attribution

Data © respective Kaggle authors (see dataset page). Code in this repository: MIT (or your choice — add `LICENSE` if you wish).