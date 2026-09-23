# Healthcare AI: From Raw Data to a Live Streamlit App

**Bootcamp Week 5 · Applications & Deployment**

## Live app

**[Add your live Streamlit Community Cloud URL here once deployed]**

## Overview

The most complete project in this portfolio: takes a clinical dataset all the way from raw data to a deployed, three-tab web application with an interactive model card, not just a trained model in a notebook. Every design decision in it, the metric used to pick a model, which features the app asks for, where the decision threshold lives, is treated as a clinical choice and explained as one, not left as a scikit-learn default.

## Dataset

Wisconsin Diagnostic Breast Cancer dataset (scikit-learn built-in), re-encoded so `1 = malignant`, the class being screened for. Eight of the thirty available features are used in the app itself, chosen for being clinically interpretable and quick for a user to enter.

## What this project demonstrates

- Framing the problem before touching data: who uses the output, what an error costs, and why that makes **recall** (not accuracy) the metric that matters here
- Exploratory analysis: class balance, per-class distribution separation, and a feature-correlation check for redundancy
- Training and comparing three models (Logistic Regression, Random Forest, SVM) inside a single `Pipeline` object, so the saved artifact always scales inputs exactly the way it was trained, the most common silent deployment bug
- Full evaluation: confusion matrix, ROC curve, classification report, and a decision-threshold table showing the real trade-off between missed malignancies and false alarms at different thresholds
- Permutation-based feature importance, model-agnostic across all three algorithm types
- Saving three deployment artifacts (`model.pkl`, `metadata.json`, `sample_patients.csv`) and verifying the reload is bit-for-bit identical to the in-notebook model before trusting it
- A three-tab Streamlit app: single-patient prediction with a risk band, batch CSV scoring with a downloadable template, and a live model card (metrics, feature importance, intended use, limitations)
- A 7-point systematic test suite (T1–T7) run against the *saved* artifact, exactly as the deployed app loads it, covering known cases, determinism, boundary values, column-order robustness, and latency
- Packaging everything for permanent deployment on Streamlit Community Cloud

## Libraries

`streamlit`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`

## Why this project is here

This is the difference between "I trained a model" and "I can ship something a clinician could actually open." The decision-threshold analysis and the model card in particular are the parts that map most directly onto real clinical AI deployment, where 0.5 is never actually the right cutoff and where a model shipped without documented limitations is a liability, not a product.

## Files

- `notebook.ipynb` — the full walkthrough, from raw data to a packaged deployment, run with saved outputs (EDA plots, ROC curve, permutation importance)
- `app.py` — the standalone three-tab Streamlit app, ready to deploy as-is
- `model.pkl` — the trained Pipeline (scaler + best model) the app loads
- `metadata.json` — feature order, input ranges, metrics, and importances, so the app never hardcodes them
- `sample_patients.csv` — three real test-set cases (benign, malignant, borderline) used by the app's example-case loader
- `requirements.txt` — dependencies for deployment

## Running it yourself

**Locally**, after cloning this repo:
```
pip install -r requirements.txt
streamlit run app.py
```

**From Colab**, open `notebook.ipynb` and run all cells — Section 8 starts the app and opens a temporary public tunnel; that link dies with the Colab session, so it's not meant to be permanent (see Section 10 in the notebook, or deploy on Streamlit Community Cloud for a link that stays live).
