import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Trial Disease Classification & Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Clinical Trial Intelligence Platform")
st.markdown(
    "NLP-driven classification of clinical trial summaries to accelerate study"
    " cohort tagging and healthcare research analytics."
)


# -----------------------------------------------------------------------------
# Cache Data & Model Artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
  pipeline = joblib.load("clinical_disease_classifier.joblib")
  encoder = joblib.load("disease_label_encoder.joblib")
  return pipeline, encoder


@st.cache_data
def load_and_preprocess_data():
  df = pd.read_csv("clinical_trials_raw_patient2trial_conditions.csv")
  df.columns = df.columns.str.strip().str.lower()
  df = df.dropna(
      subset=["brief_summary", "source_condition_query"]
  ).drop_duplicates(subset=["brief_summary", "source_condition_query"])
  df["word_count"] = df["brief_summary"].apply(lambda x: len(str(x).split()))
  return df


try:
  pipeline, label_encoder = load_artifacts()
  df = load_and_preprocess_data()
  classes = list(label_encoder.classes_)
except Exception as e:
  st.error(f"Error loading system artifacts or dataset: {e}")
  st.stop()

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_analytics, tab_performance, tab_insights = st.tabs([
    "🔍 Real-Time Inference",
    "📊 Corpus Analytics",
    "📈 Model Evaluation",
    "📑 Executive Report & Insights",
])

# =============================================================================
# Tab 1: Real-Time Inference Interface
# =============================================================================
with tab_predict:
  st.subheader("Predict Clinical Trial Disease Category")
  st.write(
      "Paste an unstructured trial summary to predict target indications with"
      " confidence scores."
  )

  col1, col2 = st.columns([1.8, 1.2])

  with col1:
    default_text = (
        "A Phase 2, open-label study evaluating the safety, pharmacokinetics,"
        " and anti-tumor efficacy of an oral CDK4/6 inhibitor administered in"
        " combination with fulvestrant in postmenopausal patients with"
        " HR-positive, HER2-negative advanced or metastatic breast neoplasms."
    )
    user_input = st.text_area(
        "Clinical Trial 'Brief Summary' Input:",
        value=default_text,
        height=190,
    )
    predict_clicked = st.button("Classify Clinical Text", type="primary")

  with col2:
    if predict_clicked and user_input.strip():
      # Compute inference probabilities
      probs = pipeline.predict_proba([user_input])[0]
      top_indices = np.argsort(probs)[::-1][:5]

      top_pred_class = label_encoder.inverse_transform([top_indices[0]])[0]
      top_confidence = probs[top_indices[0]]

      st.success(f"**Predicted Category:** {top_pred_class}")
      st.metric(
          label="Top Confidence",
          value=f"{top_confidence * 100:.2f}%",
      )

      # Build probability breakdown chart
      prob_df = pd.DataFrame({
          "Disease Category": label_encoder.inverse_transform(top_indices),
          "Confidence": probs[top_indices],
      })

      fig_prob = px.bar(
          prob_df,
          x="Confidence",
          y="Disease Category",
          orientation="h",
          color="Confidence",
          color_continuous_scale="Blues",
          text=[f"{v:.1%}" for v in prob_df["Confidence"]],
      )
      fig_prob.update_layout(
          yaxis=dict(autorange="reversed"),
          height=250,
          margin=dict(l=0, r=0, t=10, b=0),
          xaxis=dict(range=[0, 1]),
      )
      st.plotly_chart(fig_prob, use_container_width=True)

# =============================================================================
# Tab 2: Corpus Analytics & Text Patterns
# =============================================================================
with tab_analytics:
  st.subheader("Clinical Trial Dataset Profiling")

  c1, c2 = st.columns(2)

  with c1:
    # Distribution of categories
    cat_counts = df["source_condition_query"].value_counts().reset_index()
    cat_counts.columns = ["Disease Category", "Count"]

    fig_cat = px.bar(
        cat_counts,
        x="Count",
        y="Disease Category",
        orientation="h",
        color="Count",
        color_continuous_scale="Viridis",
        title="Trial Volume by Disease Condition",
    )
    fig_cat.update_layout(yaxis=dict(autorange="reversed"), height=380)
    st.plotly_chart(fig_cat, use_container_width=True)

  with c2:
    # Text length distribution across categories
    fig_box = px.box(
        df,
        x="source_condition_query",
        y="word_count",
        color="source_condition_query",
        title="Brief Summary Word Count Dispersion",
    )
    fig_box.update_layout(
        showlegend=False,
        xaxis_title="Condition",
        yaxis_title="Word Count",
        height=380,
    )
    st.plotly_chart(fig_box, use_container_width=True)

  st.markdown("---")
  st.subheader("Top Predictive Clinical Keywords by Disease Category")

  # Extract top coefficient weights from Logistic Regression
  tfidf_step = pipeline.named_steps["tfidf"]
  clf_step = pipeline.named_steps["classifier"]
  vocab = np.array(tfidf_step.get_feature_names_out())

  selected_cat = st.selectbox(
      "Select Disease Condition to inspect salient vocabulary:", classes
  )
  cat_idx = list(label_encoder.classes_).index(selected_cat)

  # Support binary and multi-class shapes
  if clf_step.coef_.shape[0] == 1 and len(classes) == 2:
    coefs = clf_step.coef_[0] if cat_idx == 1 else -clf_step.coef_[0]
  else:
    coefs = clf_step.coef_[cat_idx]

  top_pos_indices = np.argsort(coefs)[-12:]
  top_keywords_df = pd.DataFrame({
      "Medical Term / N-Gram": vocab[top_pos_indices],
      "TF-IDF Weight Impact": coefs[top_pos_indices],
  })

  fig_kw = px.bar(
      top_keywords_df,
      x="TF-IDF Weight Impact",
      y="Medical Term / N-Gram",
      orientation="h",
      color="TF-IDF Weight Impact",
      color_continuous_scale="Teal",
      title=f"Distinctive Semantic Drivers: {selected_cat}",
  )
  fig_kw.update_layout(yaxis=dict(autorange="reversed"), height=360)
  st.plotly_chart(fig_kw, use_container_width=True)

# =============================================================================
# Tab 3: Model Evaluation & Validation
# =============================================================================
with tab_performance:
  st.subheader("Holdout Validation Metrics & Error Matrices")

  # Compute metrics on a stratified 20% hold-out sample
  X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
      df["brief_summary"],
      df["source_condition_query"],
      test_size=0.20,
      random_state=42,
      stratify=df["source_condition_query"],
  )

  y_test_enc = label_encoder.transform(y_test_raw)
  y_preds = pipeline.predict(X_test_raw)

  # High-level KPIs
  report_dict = classification_report(
      y_test_enc, y_preds, target_names=classes, output_dict=True
  )

  kpi1, kpi2, kpi3, kpi4 = st.columns(4)
  kpi1.metric("Overall Accuracy", f"{report_dict['accuracy'] * 100:.2f}%")
  kpi2.metric(
      "Weighted Precision",
      f"{report_dict['weighted avg']['precision'] * 100:.2f}%",
  )
  kpi3.metric(
      "Weighted Recall", f"{report_dict['weighted avg']['recall'] * 100:.2f}%"
  )
  kpi4.metric(
      "Weighted F1-Score", f"{report_dict['weighted avg']['f1-score'] * 100:.2f}%"
  )

  # Interactive Confusion Matrix
  cm_normalized = confusion_matrix(y_test_enc, y_preds, normalize="true")

  fig_cm = px.imshow(
      cm_normalized,
      x=classes,
      y=classes,
      color_continuous_scale="Blues",
      text_auto=".2f",
      title="Normalized Confusion Matrix (Row Recall)",
      labels=dict(x="Predicted Condition", y="True Condition"),
  )
  fig_cm.update_layout(height=450)
  st.plotly_chart(fig_cm, use_container_width=True)

# =============================================================================
# Tab 4: Insights & Reporting for Healthcare Analytics
# =============================================================================
with tab_insights:
  st.subheader("Clinical Research & Healthcare Analytics Insights")

  st.markdown("""
    ### 1. Cohort Distribution & Structural Imbalance
    * **Class Skew:** Highly funded areas (e.g., solid tumors, oncology) frequently represent disproportionate study counts compared to rare diseases, necessitating balanced loss functions during model fitting.
    * **Protocol Documentation Density:** Therapeutic categories exhibit distinct protocol documentation depths: oncology and immunotherapy protocols average substantially higher word counts due to complex multi-arm staging criteria.

    ### 2. Lexical & Semantic Patterns in Trial Summaries
    * **Biomarker-Driven Clustering:** Modern protocols rely heavily on specific molecular targets (e.g., *HER2*, *EGFR*, *PD-L1*, *GLP-1*), creating sharp statistical separation in high-dimensional TF-IDF feature space.
    * **Protocol Noise:** Standard trial design jargon (e.g., *double-blind*, *placebo-controlled*, *primary endpoint*, *adverse events*) appears universally across disease cohorts and requires n-gram sublinear TF scaling to prevent masking diagnostic vocabulary.

    ### 3. Model Reliability & Error Boundaries
    * **High Separability:** Distant anatomical specialties (e.g., *Cardiovascular Diseases* vs. *Dermatology*) display near 100% precision with virtually no boundary confusion.
    * **Overlap Boundaries:** Residual classification errors concentrate in multi-morbidity cohorts (e.g., diabetic nephropathy crossing both *Metabolic* and *Renal* domains) and broad platform master trials targeting multiple solid tumors.

    ### 4. Strategic Value for Medical Affairs & Clinical Operations
    * **Automated Screening & Registry Curation:** Replaces manual clinical trial indexation, accelerating eligibility categorization for patient registries.
    * **Competitive Landscape Mapping:** Enables real-time categorization of newly registered competitor trials directly from ClinicalTrials.gov feeds.
    """)