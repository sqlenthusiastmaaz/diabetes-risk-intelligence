import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import base64
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# ═══════════════════════════ Page Configuration ═══════════════════════════
st.set_page_config(
    page_title="Diabetes Risk Intelligence | ML Dashboard",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════ Custom CSS ═══════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
.main .block-container { padding-top: 1rem; max-width: 1200px; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0c2340 0%, #1b4965 50%, #2a6f97 100%);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 6px 24px rgba(12, 35, 64, 0.3);
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.hero-content {
    max-width: 80%;
}
.hero h1 {
    font-size: 2rem; font-weight: 700;
    margin: 0; letter-spacing: -0.5px;
}
.hero p {
    margin: 0.5rem 0 0; font-size: 1rem;
    opacity: 0.8; font-weight: 400;
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
}
.hero-icon {
    font-size: 3rem;
    opacity: 0.15;
}

/* ── Gauge Cards ── */
.m-grid {
    display: flex; gap: 1rem;
    flex-wrap: wrap; margin: 1.5rem 0;
}
.m-card {
    flex: 1; min-width: 160px;
    background-color: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease;
}
.m-card:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
}
.m-card .gauge-wrap {
    display: flex; justify-content: center;
    margin-bottom: 0.3rem;
}
.m-card .val {
    font-size: 0.95rem; font-weight: 700;
    margin-top: -0.2rem;
}
.m-card .lbl {
    font-size: 0.68rem; color: var(--text-color);
    font-weight: 600; text-transform: uppercase;
    opacity: 0.55;
    letter-spacing: 0.8px; margin-top: 0.2rem;
}

/* ── Insight Box ── */
.insight {
    background-color: var(--secondary-background-color);
    border-left: 3px solid #2d4a6f;
    padding: 1.2rem 1.5rem;
    border-radius: 6px;
    margin: 1rem 0;
}
.insight h4 { margin: 0 0 0.4rem; font-weight: 600; font-size: 0.95rem; }
.insight p { margin: 0; opacity: 0.8; line-height: 1.6; font-size: 0.9rem; }

/* ── Section Header ── */
.sec-hdr {
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.5px;
    opacity: 0.6; margin-bottom: 0.5rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background-color: transparent;
    padding: 4px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; font-weight: 600;
    padding-left: 1.2rem; padding-right: 1.2rem;
}

/* ── Dividers ── */
hr {
    border: none; height: 1px;
    background: var(--secondary-background-color);
    margin: 1.5rem 0;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════ Constants ═══════════════════════════
TARGET_COL = "Diabetes_binary"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

METRIC_ICONS = {
    "Accuracy": "", "AUC": "", "Precision": "",
    "Recall": "", "F1 Score": "", "MCC": "",
}

MODEL_OBS = {
    "Logistic Regression": (
        "Best trade-off between detecting diabetes and overall discrimination. "
        "Highest AUC and MCC due to balanced class weights — ideal for medical "
        "screening where missing a diabetic case is far more costly than a false alarm."
    ),
    "Decision Tree": (
        "Prone to overfitting. Limiting tree depth controls variance but "
        "generalization remains weaker than its ensemble counterpart (Random Forest)."
    ),
    "K-Nearest Neighbors": (
        "High accuracy masks poor minority-class detection. Sensitive to class "
        "imbalance and the curse of dimensionality, mostly predicting the majority class."
    ),
    "Gaussian Naive Bayes": (
        "Surprisingly competitive despite the naive conditional independence assumption. "
        "Decent AUC score, though precision on the diabetic class is limited."
    ),
    "Random Forest": (
        "Highest raw accuracy but struggles with recall. The ensemble minimizes "
        "false positives yet misses many diabetic cases even with balanced class weights."
    ),
}


# ═══════════════════════════ Data & Model Loading ═══════════════════════════
@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load a CSV file (path or uploaded file) into a DataFrame."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


@st.cache_data
def load_metrics_csv() -> pd.DataFrame:
    """Load the pre-computed metrics comparison table."""
    try:
        return pd.read_csv(MODEL_DIR / "metrics_comparison.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model(name: str):
    """Load a pickled model from the model/ directory."""
    try:
        return joblib.load(MODEL_DIR / MODEL_FILES[name])
    except Exception as e:
        st.error(f"Error loading model '{name}': {e}")
        return None


@st.cache_resource
def load_preprocessor():
    """Load the saved StandardScaler preprocessor."""
    try:
        return joblib.load(MODEL_DIR / "preprocessor.pkl")
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None


# ═══════════════════════════ Sidebar ═══════════════════════════
with st.sidebar:
    st.markdown("## Diabetes Risk Intelligence")
    st.caption("AI-powered diabetes risk assessment")
    st.divider()

    # --- CSV Upload ---
    uploaded = st.file_uploader(
        "Upload Test CSV",
        type=["csv"],
        help=(
            "Upload a CSV with the same schema as the training data. "
            "Must include the Diabetes_binary target column."
        ),
    )

    if uploaded is not None:
        df = load_data(uploaded)
        st.success("Custom data loaded")
    elif (BASE_DIR / "test_data.csv").exists():
        df = load_data(BASE_DIR / "test_data.csv")
        st.info("Using bundled test_data.csv")
    else:
        st.error("No data available. Upload a CSV.")
        df = pd.DataFrame()

    st.divider()

    # --- Model Selector ---
    selected = st.selectbox(
        "Prediction Model",
        list(MODEL_FILES.keys()),
        help="Choose a pre-trained ML model to evaluate on the loaded dataset.",
    )

    st.divider()

    # --- Dataset Quick Stats ---
    st.markdown("**Dataset Overview**")
    if not df.empty:
        st.markdown(f"- **Rows:** {len(df):,}")
        st.markdown(f"- **Features:** {df.shape[1] - 1}")
        if TARGET_COL in df.columns:
            pos = int(df[TARGET_COL].sum())
            neg = len(df) - pos
            st.markdown(f"- **Positive (1):** {pos} ({pos/len(df)*100:.1f}%)")
            st.markdown(f"- **Negative (0):** {neg} ({neg/len(df)*100:.1f}%)")
    else:
        st.markdown("_No data loaded._")


# ═══════════════════════════ Validation ═══════════════════════════
if df.empty:
    st.stop()

if TARGET_COL not in df.columns:
    st.error(f"Dataset must contain the target column: `{TARGET_COL}`")
    st.stop()

model = load_model(selected)
preprocessor = load_preprocessor()

if model is None or preprocessor is None:
    st.warning(
        "Models or preprocessor not available. "
        "Please run `python model/train_models.py` first."
    )
    st.stop()


# ═══════════════════════════ Predictions ═══════════════════════════
X_raw = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

try:
    X_proc = pd.DataFrame(
        preprocessor.transform(X_raw),
        columns=X_raw.columns,
        index=X_raw.index,
    )
except Exception as e:
    st.error(f"Preprocessing error: {e}. Ensure your CSV schema matches the training data.")
    st.stop()

y_pred = model.predict(X_proc)
y_prob = (
    model.predict_proba(X_proc)[:, 1]
    if hasattr(model, "predict_proba")
    else y_pred.astype(float)
)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

metrics = {
    "Accuracy": acc, "Area Under the Curve": auc, "Precision": prec,
    "Recall": rec, "F1 Score": f1, "Matthews Correlation Coefficient": mcc,
}

# ═══════════════════════════ Chart Theme ═══════════════════════════
CHART_COLORS = {
    "primary": "#0f3460",
    "accent": "#e94560",
    "purple": "#533483",
    "palette": ["#0f3460", "#533483", "#e94560", "#4cc9f0", "#f4a261"],
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


# ═══════════════════════════ Hero Banner ═══════════════════════════
def get_base64_image(image_path):
    if image_path.exists():
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_b64 = get_base64_image(BASE_DIR / "logo.png")
if not logo_b64:
    logo_b64 = get_base64_image(BASE_DIR / "logo.jpg")
    img_type = "jpeg"
else:
    img_type = "png"

if logo_b64:
    hero_icon_html = f'<img src="data:image/{img_type};base64,{logo_b64}" class="hero-img">'
else:
    hero_icon_html = '<div class="hero-icon">🩺</div>'

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-content">
            <h1>Diabetes Risk Analytics</h1>
            <p>Predict diabetes risk using ML models trained on CDC BRFSS 2015 survey data</p>
            <span class="tag">Active model: {selected}</span>
        </div>
        {hero_icon_html}
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════ Tabs ═══════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Model Performance",
    "Data Explorer",
    "Risk Factors",
])


# ────────────────── TAB 1: Overview ──────────────────
with tab1:

    # --- Gauge charts ---
    GAUGE_COLORS = {
        "Accuracy": "#4DA3D9",
        "Area Under the Curve": "#4DA3D9",
        "Precision": "#4DA3D9",
        "Recall": "#4DA3D9",
        "F1 Score": "#4DA3D9",
        "Matthews Correlation Coefficient": "#4DA3D9",
    }

    def make_gauge_svg(value, label, color, size=120):
        """Generate an SVG donut gauge like Power BI."""
        r = 48
        cx, cy = size // 2, size // 2
        circumference = 2 * 3.14159 * r
        # Clamp value between 0 and 1 for the arc
        pct = max(0, min(1, value))
        filled = circumference * pct
        gap = circumference - filled
        track_color = "rgba(255,255,255,0.06)"
        return f'''
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                    stroke="{track_color}" stroke-width="8"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
                    stroke="{color}" stroke-width="8"
                    stroke-dasharray="{filled:.1f} {gap:.1f}"
                    stroke-dashoffset="{circumference * 0.25:.1f}"
                    stroke-linecap="round"
                    style="transition: stroke-dasharray 0.6s ease;"/>
            <text x="{cx}" y="{cy + 1}" text-anchor="middle" dominant-baseline="central"
                  fill="currentColor" font-size="18" font-weight="700" font-family="Inter, sans-serif">
                {value:.3f}
            </text>
        </svg>
        '''

    cards_html = '<div class="m-grid">'
    for name, val in metrics.items():
        color = GAUGE_COLORS.get(name, "#2a6f97")
        svg = make_gauge_svg(val, name, color)
        cards_html += (
            f'<div class="m-card">'
            f'  <div class="gauge-wrap">{svg}</div>'
            f'  <div class="lbl">{name}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # --- Model observation ---
    obs = MODEL_OBS.get(selected, "")
    if obs:
        st.markdown(
            f"""
            <div class="insight">
                <h4>{selected} — Key Insight</h4>
                <p>{obs}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Global Comparison Table ---
    st.markdown(
        '<div class="sec-hdr">Model Comparison (All Models on Test Set)</div>',
        unsafe_allow_html=True,
    )

    comp = load_metrics_csv()
    if not comp.empty:
        numeric_cols = comp.columns[1:]
        styled = (
            comp.style
            .highlight_max(subset=numeric_cols, axis=0, props="background-color: #d4edda; color: #000000;")
            .highlight_min(subset=numeric_cols, axis=0, props="background-color: #f8d7da; color: #000000;")
            .format({c: "{:.4f}" for c in numeric_cols})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- Grouped bar chart ---
        comp_melted = comp.melt(
            id_vars=["ML Model Name"], var_name="Metric", value_name="Score"
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=comp_melted, x="Metric", y="Score",
            hue="ML Model Name", ax=ax,
            palette=CHART_COLORS["palette"],
        )
        ax.set_title("Model Metrics Comparison")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left",
            frameon=False, fontsize=9,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Global metrics comparison file not found. Run the training script to generate it.")


# ────────────────── TAB 2: Model Performance ──────────────────
with tab2:
    st.markdown(
        f'<div class="sec-hdr">{selected} — Detailed Analysis</div>',
        unsafe_allow_html=True,
    )

    col_cm, col_roc = st.columns(2)

    # --- Confusion Matrix ---
    with col_cm:
        st.subheader("Prediction Accuracy")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="PuBuGn", ax=ax, cbar=False,
            annot_kws={"size": 16, "weight": "bold"},
            linewidths=2, linecolor="white",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Prediction Accuracy")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- ROC Curve ---
    with col_roc:
        st.subheader("ROC Performance")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.fill_between(fpr, tpr, alpha=0.12, color=CHART_COLORS["purple"])
        ax.plot(
            fpr, tpr, color=CHART_COLORS["purple"], lw=2.5,
            label=f"{selected} (AUC = {auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random Baseline")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Performance")
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # --- Classification Report ---
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_true, y_pred, output_dict=True,
        target_names=["No Diabetes", "Diabetes"],
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True,
    )


# ────────────────── TAB 3: Data Explorer ──────────────────
with tab3:
    st.markdown(
        '<div class="sec-hdr">Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    # --- Quick stat pills ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Positive Class", f"{int(y_true.sum()):,}")
    imb_ratio = int((len(df) - y_true.sum()) / max(y_true.sum(), 1))
    c4.metric("Imbalance Ratio", f"1 : {imb_ratio}")

    st.divider()

    col_data, col_dist = st.columns([3, 2])

    with col_data:
        st.subheader("Data Preview")
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)

    with col_dist:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = [CHART_COLORS["primary"], CHART_COLORS["accent"]]
        counts = df[TARGET_COL].value_counts().sort_index()
        bars = ax.bar(
            ["No Diabetes (0)", "Diabetes (1)"],
            counts.values,
            color=colors, width=0.5,
            edgecolor="white", linewidth=2, zorder=3,
        )
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.02,
                f"{count:,}", ha="center", fontweight="bold", fontsize=11,
            )
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Feature Statistics")
    st.dataframe(
        df.describe().T.style.format("{:.2f}"),
        use_container_width=True,
    )


# ────────────────── TAB 4: Risk Factors ──────────────────
with tab4:
    st.markdown(
        f'<div class="sec-hdr">{selected} — Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    if selected in ("Random Forest", "Decision Tree"):
        importances = model.feature_importances_
        feat_df = (
            pd.DataFrame({"Feature": X_raw.columns, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(15)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            feat_df["Feature"], feat_df["Importance"],
            color=sns.color_palette("viridis", len(feat_df)),
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 Features — {selected}")
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=9, fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif selected == "Logistic Regression":
        coeffs = model.coef_[0]
        top_idx = np.argsort(np.abs(coeffs))[::-1][:15]
        feat_df = (
            pd.DataFrame({
                "Feature": X_raw.columns[top_idx],
                "Coefficient": coeffs[top_idx],
            })
            .sort_values("Coefficient")
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            CHART_COLORS["accent"] if c < 0 else CHART_COLORS["primary"]
            for c in feat_df["Coefficient"]
        ]
        ax.barh(feat_df["Feature"], feat_df["Coefficient"], color=colors)
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Top 15 Feature Coefficients — Logistic Regression")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            """
            <div class="insight">
                <h4>How to Read This</h4>
                <p>
                    <b>Positive coefficients</b> (blue) increase the predicted probability of
                    diabetes. <b>Negative coefficients</b> (red) decrease risk.
                    Magnitude indicates feature influence strength.
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


# ═══════════════════════════ Footer ═══════════════════════════
st.divider()
st.markdown(
    """
    <div style="text-align:center; padding:1.2rem 1rem; margin-top:1rem;
                border-top: 1px solid rgba(255,255,255,0.1);
                font-size:0.82rem; opacity:0.7;">
        Built with Streamlit &nbsp;·&nbsp; Dataset: CDC BRFSS 2015
        &nbsp;·&nbsp; BITS Pilani ML Assignment by <strong>2025ac05116</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
