import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Mass Screening Assistant",
                   page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------
# 1. Load artifacts once per session
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    meta = json.load(open("metadata.json"))
    return model, meta

model, META = load_artifacts()
FEATURES = META["features"]
CLASS_NAMES = META["target_names"]

@st.cache_data
def load_examples():
    try:
        return pd.read_csv("sample_patients.csv")
    except Exception:
        return None

examples = load_examples()

# ----------------------------------------------------------------------
# 2. Header
# ----------------------------------------------------------------------
st.title("Breast Mass Screening Assistant")
st.caption(
    f"Model: {META['model_name']}  |  trained on the Wisconsin Diagnostic "
    f"Breast Cancer dataset  |  test ROC-AUC {META['metrics']['roc_auc']:.3f}"
)
st.warning(
    "Educational demonstration only. This tool does not diagnose disease and "
    "must not be used for clinical decisions.",
    icon="⚠️",
)

tab_single, tab_batch, tab_model = st.tabs(
    ["Single patient", "Batch (CSV)", "About the model"]
)

# ----------------------------------------------------------------------
# 3. Sidebar: inputs
# ----------------------------------------------------------------------
st.sidebar.header("Measurements")

preset = "Type values manually"
if examples is not None:
    preset = st.sidebar.selectbox(
        "Load an example case", ["Type values manually"] + examples["label"].tolist()
    )

defaults = {}
for f in FEATURES:
    lo, hi, med = META["feature_ranges"][f]
    defaults[f] = med
if examples is not None and preset != "Type values manually":
    row = examples[examples["label"] == preset].iloc[0]
    for f in FEATURES:
        defaults[f] = float(row[f])

values = {}
for f in FEATURES:
    lo, hi, med = META["feature_ranges"][f]
    step = float((hi - lo) / 200)
    values[f] = st.sidebar.slider(
        f.replace("mean ", "Mean ").title(),
        min_value=float(lo), max_value=float(hi),
        value=float(defaults[f]), step=step, key=f"{preset}_{f}",
    )

st.sidebar.divider()
threshold = st.sidebar.slider(
    "Decision threshold", 0.05, 0.95, float(META["threshold"]), 0.05,
    help="Lower the threshold to catch more positives at the cost of more false alarms.",
)

# ----------------------------------------------------------------------
# 4. Single prediction
# ----------------------------------------------------------------------
def predict(frame: pd.DataFrame):
    frame = frame[FEATURES]                 # enforce training column order
    proba = model.predict_proba(frame)[:, 1]
    return proba

with tab_single:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Entered values")
        st.dataframe(
            pd.DataFrame({"Feature": FEATURES,
                          "Value": [round(values[f], 3) for f in FEATURES]}),
            hide_index=True)

    with right:
        st.subheader("Result")
        if st.button("Run prediction", type="primary"):
            row = pd.DataFrame([values])
            p = float(predict(row)[0])
            label = CLASS_NAMES[1] if p >= threshold else CLASS_NAMES[0]

            if p >= threshold:
                st.error(f"Flagged: {label}")
            else:
                st.success(f"Not flagged: {label}")

            st.metric("Probability of " + CLASS_NAMES[1], f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))

            band = ("Low" if p < 0.25 else
                    "Intermediate" if p < 0.60 else "High")
            st.write(f"**Risk band:** {band}  ·  **Threshold in use:** {threshold:.2f}")

            st.caption(
                "The probability, not the label, is the useful output. A case at "
                f"{p:.1%} sits { 'far from' if abs(p - threshold) > 0.25 else 'close to' } "
                "the decision boundary."
            )
        else:
            st.info("Set the measurements in the sidebar, then click Run prediction.")

# ----------------------------------------------------------------------
# 5. Batch prediction
# ----------------------------------------------------------------------
with tab_batch:
    st.subheader("Score many patients at once")
    st.write("Upload a CSV containing these columns:")
    st.code(", ".join(FEATURES), language="text")

    template = pd.DataFrame([{f: META["feature_ranges"][f][2] for f in FEATURES}])
    st.download_button("Download a blank template",
                       template.to_csv(index=False).encode(),
                       "template.csv", "text/csv")

    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            batch = pd.read_csv(up)
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                probs = predict(batch)
                out = batch.copy()
                out["probability"] = probs.round(4)
                out["prediction"] = np.where(probs >= threshold,
                                             CLASS_NAMES[1], CLASS_NAMES[0])
                st.dataframe(out)
                st.download_button("Download results",
                                   out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
                st.bar_chart(out["prediction"].value_counts())
        except Exception as e:
            st.error(f"Could not read that file: {e}")

# ----------------------------------------------------------------------
# 6. Model card
# ----------------------------------------------------------------------
with tab_model:
    st.subheader("Model card")
    m = META["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Recall", f"{m['recall']:.3f}")
    c3.metric("Precision", f"{m['precision']:.3f}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.write("**Feature importance** (permutation importance on the held-out test set)")
    imp = pd.Series(META["importance"]).sort_values(ascending=True)
    st.bar_chart(imp)

    st.markdown(
        f"""
**Training data** — Wisconsin Diagnostic Breast Cancer dataset
({META['n_samples']} biopsies, {META['n_positive']} malignant). Features are computed
from digitised images of a fine needle aspirate.

**Intended use** — classroom demonstration of an end-to-end ML deployment.

**Out of scope** — real patients, other imaging modalities, populations unlike the
original cohort (Wisconsin, 1990s).

**Known limitations** — small single-centre dataset; no external validation; the
{META['threshold']} default threshold is a statistical convention, not a clinical one.

**No data retention** — values you enter are held in session memory and are never
written to disk or logged.
"""
    )
