import base64
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

TARGET_COL = "Diabetes_binary"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

# Final experiment:
# Original dataset: 253,680
# Stratified working sample: 50,000
# Train: 40,000
# Test: 10,000
WORKING_SAMPLE_SIZE = 50_000
TRAIN_ROWS = 40_000
TEST_ROWS = 10_000
RANDOM_STATE = 42


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Intelligence | ML Dashboard",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# Keep the original visual design, but force the main content
# to use the available browser width.
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1.5rem;
    width: 100%;
    max-width: none;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #0c2340 0%, #1b4965 50%, #2a6f97 100%);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 6px 24px rgba(12, 35, 64, 0.30);
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 250px;
}

.hero-content {
    max-width: 68%;
    z-index: 2;
}

.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.hero p {
    margin: 0.5rem 0 0;
    font-size: 1rem;
    opacity: 0.8;
    font-weight: 400;
}

.hero .tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    padding: 0.3rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin-top: 1rem;
    border: 1px solid rgba(255,255,255,0.2);
}

.hero-img {
    height: 20vw;
    max-height: 220px;
    min-height: 120px;
    width: auto;
    border-radius: 12px;
    object-fit: contain;
    opacity: 0.9;
    margin-right: 1rem;
}

.hero-icon {
    font-size: 3rem;
    opacity: 0.15;
}

/* Gauge Cards */
.m-grid {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}

.m-card {
    flex: 1;
    min-width: 145px;
    background-color: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.m-card .gauge-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 0.3rem;
}

.m-card .lbl {
    font-size: 0.68rem;
    color: var(--text-color);
    font-weight: 600;
    text-transform: uppercase;
    opacity: 0.55;
    letter-spacing: 0.8px;
    margin-top: 0.2rem;
}

/* Insight Box */
.insight {
    background-color: var(--secondary-background-color);
    border-left: 3px solid #2d4a6f;
    padding: 1.2rem 1.5rem;
    border-radius: 6px;
    margin: 1rem 0;
}

.insight h4 {
    margin: 0 0 0.4rem;
    font-weight: 600;
    font-size: 0.95rem;
}

.insight p {
    margin: 0;
    opacity: 0.8;
    line-height: 1.6;
    font-size: 0.9rem;
}

/* Winner Box */
.winner {
    background: linear-gradient(
        135deg,
        rgba(45, 74, 111, 0.10),
        rgba(42, 111, 151, 0.08)
    );
    border: 1px solid rgba(45, 74, 111, 0.25);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
}

.winner h3 {
    margin: 0 0 0.35rem;
    font-size: 1rem;
}

.winner p {
    margin: 0;
    opacity: 0.82;
    line-height: 1.5;
    font-size: 0.86rem;
}

/* Section Header */
.sec-hdr {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.6;
    margin-bottom: 0.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    padding: 4px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* Dividers */
hr {
    border: none;
    height: 1px;
    background: var(--secondary-background-color);
    margin: 1.5rem 0;
}

/* DataFrame */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FINAL MODEL OBSERVATIONS
# ============================================================

MODEL_OBS = {
    "Logistic Regression": (
        "Strongest overall model for this experiment. It achieved the "
        "highest AUC (0.8345), Recall (0.7782), F1 Score (0.4565), and "
        "MCC (0.3771), giving the best overall balance for this "
        "imbalanced classification problem."
    ),
    "Decision Tree": (
        "Provides relatively high recall (0.7157), but its lower AUC "
        "(0.7448), F1 Score (0.4146), and MCC (0.3173) indicate weaker "
        "overall classification performance than Logistic Regression "
        "and Random Forest."
    ),
    "K-Nearest Neighbors": (
        "Achieved the highest accuracy (0.8519) and precision (0.4345), "
        "but its recall (0.2096), F1 Score (0.2828), and MCC (0.2288) "
        "are low. This shows why accuracy alone is not sufficient for "
        "this imbalanced dataset."
    ),
    "Gaussian Naive Bayes": (
        "Achieved high accuracy (0.7798), but its lower recall (0.5793), "
        "F1 Score (0.4230), and MCC (0.3164) indicate weaker minority-"
        "class detection than the leading models."
    ),
    "Random Forest": (
        "Strong second-place model. It achieved 77.49% accuracy, "
        "0.8204 AUC, 0.6691 recall, 0.4530 F1 Score, and 0.3587 MCC. "
        "Its performance is close to Logistic Regression on F1 and MCC."
    ),
}


# ============================================================
# DATA & MODEL LOADING
# ============================================================

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as exc:
        st.error(f"Error loading CSV: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_metrics_csv():
    try:
        return pd.read_csv(MODEL_DIR / "metrics_comparison.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model(name):
    try:
        return joblib.load(MODEL_DIR / MODEL_FILES[name])
    except Exception as exc:
        st.error(f"Error loading model '{name}': {exc}")
        return None


@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load(MODEL_DIR / "preprocessor.pkl")
    except Exception as exc:
        st.error(f"Error loading preprocessor: {exc}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Diabetes Risk Intelligence")
    st.caption("AI-powered diabetes risk assessment")
    st.divider()

    uploaded = st.file_uploader(
        "Upload Test CSV",
        type=["csv"],
        help=(
            "Upload a CSV with the same raw feature schema as the "
            "training data. Diabetes_binary is optional: include it only "
            "if you want evaluation metrics."
        ),
    )

    if uploaded is not None:
        df = load_data(uploaded)
        if not df.empty:
            st.success("Custom data loaded")
    elif (BASE_DIR / "test_data.csv").exists():
        df = load_data(BASE_DIR / "test_data.csv")
        if not df.empty:
            st.info("Using bundled test_data.csv")
    else:
        df = pd.DataFrame()
        st.error("No data available. Upload a CSV.")

    st.divider()

    selected = st.selectbox(
        "Prediction Model",
        list(MODEL_FILES.keys()),
        help="Choose a pre-trained ML model to evaluate on the loaded dataset.",
    )

    st.divider()

    st.markdown("**Dataset Overview**")

    if not df.empty:
        st.markdown(f"- **Rows:** {len(df):,}")

        if TARGET_COL in df.columns:
            st.markdown(f"- **Features:** {df.shape[1] - 1}")

            pos = int(df[TARGET_COL].sum())
            neg = len(df) - pos

            st.markdown(
                f"- **Positive (1):** {pos:,} ({pos / len(df) * 100:.1f}%)"
            )
            st.markdown(
                f"- **Negative (0):** {neg:,} ({neg / len(df) * 100:.1f}%)"
            )
    else:
        st.markdown("_No data loaded._")


# ============================================================
# VALIDATION
# ============================================================

if df.empty:
    st.stop()

# The target is optional for prediction-only uploads.
# If it is present, the app also calculates evaluation metrics.
has_target = TARGET_COL in df.columns

model = load_model(selected)
preprocessor = load_preprocessor()

if model is None or preprocessor is None:
    st.warning(
        "Models or preprocessor are not available. "
        "Run `python model/train_models.py` first."
    )
    st.stop()


# ============================================================
# PREDICTION
# ============================================================

if has_target:
    X_raw = df.drop(columns=[TARGET_COL])
    y_true = df[TARGET_COL]
else:
    X_raw = df.copy()
    y_true = None

# Validate the feature schema against the fitted preprocessor.
try:
    expected_features = list(preprocessor.feature_names_in_)
except Exception:
    expected_features = list(X_raw.columns)

missing = [
    col for col in expected_features
    if col not in X_raw.columns
]

if missing:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        + ", ".join(missing)
    )
    st.stop()

# Ignore accidental extra columns, while preserving the training order.
X_raw = X_raw[expected_features]

try:
    X_proc = preprocessor.transform(X_raw)
except Exception as exc:
    st.error(
        f"Preprocessing error: {exc}. "
        "Ensure the CSV schema and value types match the training data."
    )
    st.stop()

try:
    y_pred = model.predict(X_proc)

    probability_available = hasattr(model, "predict_proba")

    if probability_available:
        y_prob = model.predict_proba(X_proc)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_proc)
    else:
        y_prob = y_pred.astype(float)

except Exception as exc:
    st.error(f"Prediction error: {exc}")
    st.stop()

# Attach predictions to the currently loaded dataset.
prediction_results = df.copy()
prediction_results["Predicted_Diabetes"] = y_pred

if probability_available:
    prediction_results["Risk_Probability"] = y_prob
    prediction_results["Risk_Level"] = np.where(
        y_prob >= 0.5,
        "Higher Risk",
        "Lower Risk",
    )
else:
    prediction_results["Prediction_Score"] = y_prob
    prediction_results["Risk_Level"] = np.where(
        y_pred == 1,
        "Higher Risk",
        "Lower Risk",
    )


# ============================================================
# CURRENT DATASET METRICS
# ============================================================

if has_target:
    acc = accuracy_score(y_true, y_pred)

    if y_true.nunique() == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    prec = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    rec = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )
else:
    acc = auc = prec = rec = f1 = mcc = np.nan

metrics = {
    "Accuracy": acc,
    "Area Under the Curve": auc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1,
    "Matthews Correlation Coefficient": mcc,
}

# ============================================================
# CHART THEME
# ============================================================

CHART_COLORS = {
    "primary": "#0f3460",
    "accent": "#e94560",
    "purple": "#533483",
    "palette": [
        "#0f3460",
        "#533483",
        "#e94560",
        "#4cc9f0",
        "#f4a261",
    ],
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlepad": 12,
    "axes.labelweight": "bold",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})


# ============================================================
# HERO BANNER
# ============================================================

def get_base64_image(image_path):
    if image_path.exists():
        try:
            return base64.b64encode(
                image_path.read_bytes()
            ).decode()
        except Exception:
            return ""
    return ""


logo_b64 = get_base64_image(BASE_DIR / "logo.png")
img_type = "png"

if not logo_b64:
    logo_b64 = get_base64_image(BASE_DIR / "logo.jpg")
    img_type = "jpeg"

if logo_b64:
    hero_icon_html = (
        f'<img src="data:image/{img_type};base64,{logo_b64}" '
        'class="hero-img">'
    )
else:
    hero_icon_html = '<div class="hero-icon">🩺</div>'

hero_html = f"""
<div class="hero">
    <div class="hero-content">
        <h1>Diabetes Risk Analytics</h1>
        <p>
            Predict diabetes risk using ML models trained on CDC BRFSS 2015 survey data
        </p>
        <span class="tag">Active model: {selected}</span>
    </div>
    {hero_icon_html}
</div>
"""

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)

if has_target:
    st.success(
        "Evaluation mode: uploaded data includes `Diabetes_binary`, "
        "so predictions and performance metrics are available."
    )
else:
    st.info(
        "Prediction-only mode: uploaded data does not include `Diabetes_binary`. "
        "Predictions will be generated, but evaluation metrics cannot be calculated."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Model Performance",
    "Data Explorer",
    "Risk Factors",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    # --------------------------------------------------------
    # Evaluation metrics OR prediction summary
    # --------------------------------------------------------

    if has_target:
        GAUGE_COLORS = {
            "Accuracy": "#4DA3D9",
            "Area Under the Curve": "#4DA3D9",
            "Precision": "#4DA3D9",
            "Recall": "#4DA3D9",
            "F1 Score": "#4DA3D9",
            "Matthews Correlation Coefficient": "#4DA3D9",
        }

        def make_gauge_svg(value, size=120):
            display_value = f"{value:.3f}" if not pd.isna(value) else "N/A"
            pct = 0 if pd.isna(value) else max(0, min(1, value))

            r = 48
            cx = size // 2
            cy = size // 2
            circumference = 2 * np.pi * r
            filled = circumference * pct
            gap = circumference - filled

            return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
        stroke="rgba(128,128,128,0.15)" stroke-width="8"/>
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
        stroke="#4DA3D9" stroke-width="8"
        stroke-dasharray="{filled:.1f} {gap:.1f}"
        stroke-dashoffset="{circumference * 0.25:.1f}"
        stroke-linecap="round"/>
    <text x="{cx}" y="{cy + 1}" text-anchor="middle"
        dominant-baseline="central" fill="currentColor"
        font-size="18" font-weight="700" font-family="Arial, sans-serif">
        {display_value}
    </text>
</svg>
"""

        cards_html = '<div class="m-grid">'

        for name, val in metrics.items():
            cards_html += (
                '<div class="m-card">'
                f'<div class="gauge-wrap">{make_gauge_svg(val)}</div>'
                f'<div class="lbl">{name}</div>'
                '</div>'
            )

        cards_html += "</div>"

        st.markdown(
            cards_html,
            unsafe_allow_html=True,
        )

        # Model-specific interpretation applies to the official experiment.
        obs = MODEL_OBS.get(selected, "")

        if obs:
            insight_html = f"""
<div class="insight">
    <h4>{selected} — Key Insight</h4>
    <p>{obs}</p>
</div>
"""
            st.markdown(
                insight_html,
                unsafe_allow_html=True,
            )

        st.info(
            "The metrics above are calculated on the currently loaded dataset. "
            "Accuracy alone is not sufficient because the target is imbalanced; "
            "Recall, F1 Score, MCC and AUC should be considered together."
        )

    else:
        # --------------------------------------------------------
        # Prediction-only mode
        # --------------------------------------------------------
        # Do NOT show six N/A evaluation gauges. Without the true target,
        # evaluation metrics are mathematically unavailable. Instead show
        # useful prediction information.
        predicted_positive = int(np.sum(y_pred == 1))
        predicted_negative = int(np.sum(y_pred == 0))

        st.markdown(
            '<div class="sec-hdr">CUSTOM DATA PREDICTION</div>',
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Total Predictions",
            f"{len(y_pred):,}",
        )

        p2.metric(
            "Higher Risk",
            f"{predicted_positive:,}",
        )

        p3.metric(
            "Lower Risk",
            f"{predicted_negative:,}",
        )

        st.info(
            "This uploaded CSV does not contain `Diabetes_binary`, so the "
            "model can generate predictions and risk probabilities, but "
            "performance metrics cannot be calculated. Add `Diabetes_binary` "
            "only when you want to evaluate predictions against known outcomes."
        )

        st.subheader("Prediction Summary")

        summary_df = pd.DataFrame(
            {
                "Prediction": [
                    "Higher Risk (1)",
                    "Lower Risk (0)",
                ],
                "Count": [
                    predicted_positive,
                    predicted_negative,
                ],
                "Percentage": [
                    predicted_positive / len(y_pred) * 100,
                    predicted_negative / len(y_pred) * 100,
                ],
            }
        )

        st.dataframe(
            summary_df.style.format(
                {"Percentage": "{:.1f}%"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Official experiment winner
    # --------------------------------------------------------
    # This is deliberately separate from the uploaded-data prediction.
    # It describes the fixed 50K -> 40K/10K experiment.

    comp = load_metrics_csv()

    if not comp.empty:
        required = {
            "ML Model Name",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
        }

        if required.issubset(comp.columns):

            winner_row = comp.loc[
                comp["MCC"].idxmax()
            ]

            winner_name = winner_row["ML Model Name"]
            winner_mcc = winner_row["MCC"]

            winner_html = f"""
<div class="winner">
    <h3>🏆 Official Experiment Winner: {winner_name}</h3>
    <p>
        {winner_name} was selected as the overall winner in the final
        common test-set experiment based on the strongest overall balance
        of AUC, Recall, F1 Score and MCC. Its MCC is
        <b>{winner_mcc:.4f}</b>.
    </p>
</div>
"""

            st.markdown(
                winner_html,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Official comparison table
    # --------------------------------------------------------

    st.markdown(
        '<div class="sec-hdr">'
        'Model Comparison — All Models on Same Test Set'
        '</div>',
        unsafe_allow_html=True,
    )

    if not comp.empty:

        numeric_cols = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
        ]

        available_numeric = [
            col for col in numeric_cols
            if col in comp.columns
        ]

        styled = (
            comp.style
            .highlight_max(
                subset=available_numeric,
                axis=0,
                props=(
                    "background-color: #d4edda;"
                    "color: #000000;"
                ),
            )
            .highlight_min(
                subset=available_numeric,
                axis=0,
                props=(
                    "background-color: #f8d7da;"
                    "color: #000000;"
                ),
            )
            .format(
                {
                    col: "{:.4f}"
                    for col in available_numeric
                }
            )
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Official experiment: 50,000-row stratified working sample → "
            "40,000 training rows + 10,000 held-out test rows. "
            "All five models used the same training and test sets."
        )

        # ----------------------------------------------------
        # Grouped comparison chart
        # ----------------------------------------------------

        comp_melted = comp.melt(
            id_vars=["ML Model Name"],
            value_vars=available_numeric,
            var_name="Metric",
            value_name="Score",
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        sns.barplot(
            data=comp_melted,
            x="Metric",
            y="Score",
            hue="ML Model Name",
            ax=ax,
            palette=CHART_COLORS["palette"],
        )

        ax.set_title(
            "Model Metrics Comparison"
        )

        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=9,
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:
        st.info(
            "Global metrics comparison file not found. "
            "Run the training script to generate it."
        )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================

with tab2:

    if not has_target:
        st.markdown(
            '<div class="sec-hdr">Prediction Results</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "This upload does not contain `Diabetes_binary`, so confusion matrix, "
            "ROC/AUC, precision, recall, F1 and MCC cannot be calculated. "
            "The model has still generated predictions for every uploaded row."
        )

        p1, p2, p3 = st.columns(3)
        predicted_positive = int(np.sum(y_pred == 1))
        predicted_negative = int(np.sum(y_pred == 0))

        p1.metric("Total Predictions", f"{len(y_pred):,}")
        p2.metric("Higher Risk", f"{predicted_positive:,}")
        p3.metric("Lower Risk", f"{predicted_negative:,}")

        st.subheader("Prediction Results")
        st.dataframe(
            prediction_results.head(100),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.markdown(
            f'<div class="sec-hdr">{selected} — Detailed Analysis</div>',
            unsafe_allow_html=True,
        )

        col_cm, col_roc = st.columns(2)

        # --------------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------------
        with col_cm:
            st.subheader("Prediction Accuracy")

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            )

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="PuBuGn",
                ax=ax,
                cbar=False,
                annot_kws={
                    "size": 16,
                    "weight": "bold",
                },
                linewidths=2,
                linecolor="white",
                xticklabels=[
                    "No Diabetes",
                    "Diabetes",
                ],
                yticklabels=[
                    "No Diabetes",
                    "Diabetes",
                ],
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Prediction Accuracy")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # --------------------------------------------------------
        # ROC
        # --------------------------------------------------------
        with col_roc:
            st.subheader("ROC Performance")

            if not pd.isna(auc):
                fpr, tpr, _ = roc_curve(
                    y_true,
                    y_prob,
                )

                fig, ax = plt.subplots(
                    figsize=(5, 4)
                )

                ax.fill_between(
                    fpr,
                    tpr,
                    alpha=0.12,
                    color=CHART_COLORS["purple"],
                )

                ax.plot(
                    fpr,
                    tpr,
                    color=CHART_COLORS["purple"],
                    lw=2.5,
                    label=f"{selected} (AUC = {auc:.3f})",
                )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    "k--",
                    lw=1,
                    alpha=0.4,
                    label="Random Baseline",
                )

                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1.05])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Performance")

                ax.legend(
                    loc="lower right",
                    frameon=True,
                )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning(
                    "ROC/AUC cannot be calculated because the loaded "
                    "test data contains only one class."
                )

        st.divider()

        # --------------------------------------------------------
        # Classification Report
        # --------------------------------------------------------
        st.subheader("Classification Report")

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No Diabetes",
                "Diabetes",
            ],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(
            report_dict
        ).transpose()

        st.dataframe(
            report_df.style.format("{:.3f}"),
            use_container_width=True,
        )

# ============================================================
# TAB 3 — DATA EXPLORER
# ============================================================

with tab3:

    st.markdown(
        '<div class="sec-hdr">Dataset & Prediction Explorer</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Rows",
        f"{len(df):,}",
    )

    feature_count = len(expected_features)
    c2.metric(
        "Features",
        f"{feature_count:,}",
    )

    predicted_positive = int(np.sum(y_pred == 1))
    predicted_negative = int(np.sum(y_pred == 0))

    c3.metric(
        "Predicted Higher Risk",
        f"{predicted_positive:,}",
    )

    c4.metric(
        "Predicted Lower Risk",
        f"{predicted_negative:,}",
    )

    st.divider()

    # --------------------------------------------------------
    # Prediction Results
    # --------------------------------------------------------
    st.subheader("Prediction Results")

    if has_target:
        st.caption(
            "The uploaded dataset contains the true `Diabetes_binary` target, "
            "so predictions and evaluation metrics are both available."
        )
    else:
        st.caption(
            "Prediction-only mode: `Diabetes_binary` was not provided. "
            "Predicted class and risk score are still generated for every row."
        )

    st.dataframe(
        prediction_results.head(100),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = prediction_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="diabetes_predictions.csv",
        mime="text/csv",
    )

    st.divider()

    # --------------------------------------------------------
    # Original Data Preview + Target Distribution
    # --------------------------------------------------------
    col_data, col_dist = st.columns([3, 2])

    with col_data:
        st.subheader("Original Data Preview")

        st.dataframe(
            df.head(15),
            use_container_width=True,
            hide_index=True,
        )

    with col_dist:
        if has_target:
            st.subheader("Actual Target Distribution")

            counts = (
                df[TARGET_COL]
                .value_counts()
                .reindex(
                    [0, 1],
                    fill_value=0,
                )
            )

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            colors = [
                CHART_COLORS["primary"],
                CHART_COLORS["accent"],
            ]

            bars = ax.bar(
                [
                    "No Diabetes (0)",
                    "Diabetes (1)",
                ],
                counts.values,
                color=colors,
                width=0.5,
                edgecolor="white",
                linewidth=2,
                zorder=3,
            )

            for bar, count in zip(
                bars,
                counts.values,
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.02,
                    f"{count:,}",
                    ha="center",
                    fontweight="bold",
                    fontsize=11,
                )

            ax.set_ylabel("Count")
            ax.set_title("Actual Class Distribution")
            ax.grid(
                axis="y",
                alpha=0.3,
                zorder=0,
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.subheader("Predicted Class Distribution")

            pred_counts = (
                pd.Series(y_pred)
                .value_counts()
                .reindex([0, 1], fill_value=0)
            )

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            colors = [
                CHART_COLORS["primary"],
                CHART_COLORS["accent"],
            ]

            bars = ax.bar(
                [
                    "Lower Risk (0)",
                    "Higher Risk (1)",
                ],
                pred_counts.values,
                color=colors,
                width=0.5,
                edgecolor="white",
                linewidth=2,
                zorder=3,
            )

            for bar, count in zip(
                bars,
                pred_counts.values,
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(pred_counts.values) * 0.02,
                    f"{count:,}",
                    ha="center",
                    fontweight="bold",
                    fontsize=11,
                )

            ax.set_ylabel("Count")
            ax.set_title("Predicted Class Distribution")
            ax.grid(
                axis="y",
                alpha=0.3,
                zorder=0,
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    st.subheader("Feature Statistics")
    st.dataframe(
        df.describe().T.style.format(
            "{:.2f}"
        ),
        use_container_width=True,
    )

# ============================================================
# TAB 4 — RISK FACTORS
# ============================================================

with tab4:

    st.markdown(
        f'<div class="sec-hdr">{selected} — Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Tree-based models
    # --------------------------------------------------------

    if selected in (
        "Random Forest",
        "Decision Tree",
    ):

        importances = model.feature_importances_

        feat_df = (
            pd.DataFrame(
                {
                    "Feature": expected_features,
                    "Importance": importances,
                }
            )
            .sort_values(
                "Importance",
                ascending=True,
            )
            .tail(15)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        bars = ax.barh(
            feat_df["Feature"],
            feat_df["Importance"],
            color=sns.color_palette(
                "viridis",
                len(feat_df),
            ),
        )

        ax.set_xlabel("Importance")
        ax.set_title(
            f"Top 15 Features — {selected}"
        )

        for bar in bars:
            width = bar.get_width()

            ax.text(
                width + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.info(
            "Feature importance indicates how much each feature "
            "contributed to the tree-based model's decisions. "
            "It represents model importance, not causation."
        )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    elif selected == "Logistic Regression":

        coeffs = model.coef_[0]

        top_idx = (
            np.argsort(
                np.abs(coeffs)
            )[::-1][:15]
        )

        feat_df = (
            pd.DataFrame(
                {
                    "Feature": np.array(expected_features)[top_idx],
                    "Coefficient": coeffs[top_idx],
                }
            )
            .sort_values(
                "Coefficient"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        colors = [
            CHART_COLORS["accent"]
            if coefficient < 0
            else CHART_COLORS["primary"]
            for coefficient
            in feat_df["Coefficient"]
        ]

        ax.barh(
            feat_df["Feature"],
            feat_df["Coefficient"],
            color=colors,
        )

        ax.set_xlabel(
            "Coefficient Value"
        )

        ax.set_title(
            "Top 15 Feature Coefficients — Logistic Regression"
        )

        ax.axvline(
            x=0,
            color="gray",
            linestyle="--",
            alpha=0.5,
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.markdown(
            """
<div class="insight">
    <h4>How to Read This</h4>
    <p>
        <b>Positive coefficients</b> increase the model's predicted
        log-odds of the positive class. <b>Negative coefficients</b>
        decrease the model's predicted log-odds of the positive class.
        Larger absolute values indicate stronger influence on the
        standardized linear decision function. These are
        <b>model associations, not causal effects</b>.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

    else:

        st.info(
            f"Native feature importance is not available for **{selected}**. "
            "Select Random Forest, Decision Tree, or Logistic Regression "
            "for feature analysis."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div style="
    text-align:center;
    padding:1.2rem 1rem;
    margin-top:1rem;
    border-top:1px solid rgba(128,128,128,0.15);
    font-size:0.82rem;
    opacity:0.7;
">
    Built with Streamlit &nbsp;·&nbsp; Dataset: CDC BRFSS 2015
    &nbsp;·&nbsp; BITS Pilani ML Assignment
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Educational ML demonstration only. This application is not a "
    "medical diagnostic tool and should not replace professional medical advice."
)