<div align="center">

# Coffee Vending Sales ML

**Machine learning prediction system**

Multiclass drink prediction from real POS transaction logs — from EDA and model comparison to a deployable prediction service.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Portfolio](https://img.shields.io/badge/Portfolio-mounir--gmamsi-1b3550?style=for-the-badge&logo=netlify&logoColor=white)](https://mounir-gmamsi.netlify.app/)

</div>

---

## Overview

This project is the midterm for the DataTalks.Club Machine Learning Zoomcamp. It predicts `coffee_name` from vending-machine transaction context to support inventory planning and operational decisions. The work covers exploration, model training and comparison, and a container-ready prediction path.

---

## What the product delivers

| Capability | Description |
|------------|-------------|
| Exploratory analysis | Understand sales patterns by time, payment type and drink mix |
| Feature engineering | Build usable features from POS logs |
| Model training | Train and compare tabular classifiers |
| Evaluation | Accuracy and macro-F1 for imbalanced classes |
| Serving | Prediction API / app path for inference |
| Packaging | Docker-oriented delivery |

---

## Technology stack

### Languages and notebooks

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

### Data and ML

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)

### Serving and packaging

![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F9D0C6?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)

| Layer | Technologies |
|-------|----------------|
| Analysis | Jupyter, Pandas, NumPy, Matplotlib, Seaborn |
| Modelling | scikit-learn, XGBoost, joblib |
| Serving | Flask, Gradio, prediction scripts |
| Packaging | Docker, requirements.txt |

---

## Solution flow

| Step | Stage | Technologies |
|:----:|-------|--------------|
| 1 | Explore POS data | Jupyter, Pandas |
| 2 | Engineer features | Python, Pandas |
| 3 | Train and compare models | scikit-learn, XGBoost |
| 4 | Export best model | joblib |
| 5 | Serve predictions | Flask / Gradio, Docker |

```mermaid
flowchart TD
  A[1. POS transaction data] --> B[2. EDA and features]
  B --> C[3. Train and compare models]
  C --> D[4. Export model]
  D --> E[5. Prediction service]
```

---

## Repository structure

| File / folder | Purpose |
|---------------|---------|
| `notebook.ipynb` | EDA, training and evaluation |
| `train.py` | Training script |
| `predict.py` / `app.py` | Inference / service entrypoints |
| `models/` | Saved model artefacts |
| `Dockerfile` | Container packaging |
| `requirements.txt` | Python dependencies |

---

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

For serving, use the prediction scripts or Docker build as described in the repository files.

---

## Author

**Mounir Gmamsi**  
[Portfolio](https://mounir-gmamsi.netlify.app/) · [GitHub](https://github.com/Mr-Gms-33) · [LinkedIn](https://www.linkedin.com/in/mounir-gmamsi-1595951b3)
