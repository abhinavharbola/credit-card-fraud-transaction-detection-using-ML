import os, json, math
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Initialize SHAP JS plots
shap.initjs()

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file with transactions to be analyzed. Now with SHAP explainability!")

# -----------------------
# Helpers
# -----------------------
def _clean_numeric_series(s):
    """
    Remove stray brackets/quotes/parentheses/whitespace and coerce to numeric.
    Keep only valid float characters.
    """
    s_str = s.astype(str) if not pd.api.types.is_string_dtype(s) else s
    # Keep only digits, '.', 'E', and '-' (for scientific notation)
    s_str = s_str.str.replace(r'[^0-9.E-]', '', regex=True)
    return pd.to_numeric(s_str, errors='coerce')

def clean_numeric_df(df, cols=None):
    """Coerce selected columns (or all) to numeric safely, fill na with median or 0."""
    df = df.copy()
    target_cols = df.columns if cols is None else cols
    for c in target_cols:
        if c not in df.columns:
            continue
        try:
            ser = _clean_numeric_series(df[c])
        except Exception:
            ser = pd.to_numeric(df[c], errors='coerce')
        
        med = ser.median()
        if pd.isna(med):
            med = 0.0
        
        ser = ser.fillna(med)
        df[c] = ser
    return df

@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["data_path"] = BASE / "creditcard.csv"
    artifacts["model_path"] = BASE / "fraud_model.pkl"
    artifacts["scaler_path"] = BASE / "scaler.pkl"
    artifacts["feature_path"] = BASE / "feature_names.json"
    artifacts["background_path"] = BASE / "shap_background.npy"

    # Load model & scaler
    if artifacts["model_path"].exists():
        artifacts["model"] = joblib.load(artifacts["model_path"])
    else:
        st.error(f"Model artifact not found at {artifacts['model_path']}. Please run train_model.py.")
        artifacts["model"] = None

    if artifacts["scaler_path"].exists():
        artifacts["scaler"] = joblib.load(artifacts["scaler_path"])
    else:
        st.error(f"Scaler artifact not found at {artifacts['scaler_path']}. Please run train_model.py.")
        artifacts["scaler"] = None

    if artifacts["feature_path"].exists():
        with open(artifacts["feature_path"], "r") as f:
            artifacts["features"] = json.load(f)
    else:
        st.error(f"Feature names not found at {artifacts['feature_path']}. Please run train_model.py.")
        artifacts["features"] = None

    # --- Load SHAP Background Data ---
    if artifacts["background_path"].exists():
        artifacts["background_data"] = np.load(artifacts["background_path"])
    else:
        st.error(f"SHAP background data not found at {artifacts['background_path']}. Please run train_model.py.")
        artifacts["background_data"] = None
    # --- End SHAP Section ---

    # Load full dataset for "Model Insights" tab
    if artifacts["data_path"].exists():
        try:
            artifacts["df"] = pd.read_csv(artifacts["data_path"])
        except Exception:
            artifacts["df"] = None
    else:
        artifacts["df"] = None

    return artifacts

# ----- SHAP Explainer Loader -----
@st.cache_resource
def load_explainer(_model, _background_data, _feature_names):
    """Create and cache the SHAP TreeExplainer."""
    if _model is None or _background_data is None or _feature_names is None:
        return None
    try:
        # Create a DataFrame for the background data with correct feature names
        background_df = pd.DataFrame(_background_data, columns=_feature_names)
        # Pass the model and the background data (as a DataFrame)
        explainer = shap.TreeExplainer(_model, background_df)
        return explainer
    except Exception as e:
        st.error(f"Failed to initialize SHAP explainer: {e}")
        return None
        
art = load_artifacts()
model = art["model"]
scaler = art["scaler"]
features = art["features"]
original_df = art["df"]

# Load the explainer
explainer = load_explainer(model, art["background_data"], features)

# -----------------------
# Main Page - Uploader
# -----------------------

with st.container(border=True):
    uploaded = st.file_uploader("Upload a CSV with transactions", type=["csv"])
    
if uploaded is None:
    st.info("Please upload a CSV file to see predictions and SHAP explanations.")

# -----------------------
# Main: prediction & visuals
# -----------------------
if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        user_df = None

    if user_df is not None:
        if model is None or scaler is None or features is None:
            st.error("App missing critical artifacts (model/scaler/features). Run train_model.py to produce them.")
        else:
            # Clean the uploaded dataframe *before* feature alignment
            user_df_cleaned = clean_numeric_df(user_df.copy(), cols=[col for col in features if col in user_df.columns])
            
            X = user_df_cleaned.copy()
            
            # Align columns to expected features
            if not set(features).issubset(set(X.columns)):
                st.warning("Uploaded file does not contain all expected feature columns. Missing columns will be filled with 0.")
                X = X.reindex(columns=features, fill_value=0)
            else:
                X = X[features] # Ensure correct order

            # We already cleaned the columns that exist, but reindex might have added new columns
            # that need to be processed (though they are filled with 0)
            # This second clean ensures all `features` columns are numeric
            X = clean_numeric_df(X, cols=features)

            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                st.error(f"Failed to scale inputs: {e}")
                st.stop()

            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            results = user_df.copy()
            results["Predicted Class"] = preds
            results["Fraud Probability"] = probs
            results["Predicted Class"] = results["Predicted Class"].map({0: "Legit", 1: "Fraud"})

            # Create tabs for results
            tab_summary, tab_data, tab_model, tab_explain = st.tabs([
                "Prediction Summary", 
                "Full Data Preview", 
                "Model Insights", 
                "SHAP Explainability"
            ])

            with tab_summary:
                st.subheader("Prediction Breakdown")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    total_trans = len(results)
                    pred_counts = results["Predicted Class"].value_counts().reset_index()
                    pred_counts.columns = ["Class", "Count"]
                    
                    # Ensure both classes are present for the chart
                    if "Fraud" not in pred_counts["Class"].values:
                        pred_counts = pd.concat([pred_counts, pd.DataFrame([{"Class": "Fraud", "Count": 0}])], ignore_index=True)
                    if "Legit" not in pred_counts["Class"].values:
                        pred_counts = pd.concat([pred_counts, pd.DataFrame([{"Class": "Legit", "Count": 0}])], ignore_index=True)
                    
                    fraud_count = int(pred_counts[pred_counts["Class"] == "Fraud"]["Count"].sum())
                    legit_count = int(pred_counts[pred_counts["Class"] == "Legit"]["Count"].sum())
                    fraud_pct = (fraud_count / total_trans) * 100 if total_trans > 0 else 0

                    fig_donut = go.Figure(go.Pie(
                        labels=pred_counts["Class"],
                        values=pred_counts["Count"],
                        hole=0.5,
                        marker_colors=["crimson" if c == "Fraud" else "mediumseagreen" for c in pred_counts["Class"]],
                        textinfo="percent+value",
                        hoverinfo="label+percent+value"
                    ))
                    fig_donut.update_layout(
                        title_text="Prediction Results",
                        annotations=[dict(text=f'{total_trans:,}<br>Total', x=0.5, y=0.5, font_size=20, showarrow=False)],
                        legend_title_text="Predicted Class",
                        showlegend=True
                    )
                    st.plotly_chart(fig_donut, width='stretch')
                
                with col2:
                    st.subheader("Summary Metrics")
                    st.metric("Total Transactions Analyzed", f"{total_trans:,}")
                    st.metric("🔴 Detected Fraud", f"{fraud_count:,}", f"{fraud_pct:.2f}% of total")
                    st.metric("🟢 Legit Transactions", f"{legit_count:,}")
                    
                    st.info(f"Analysis complete. Found **{fraud_count}** transactions predicted as 'Fraud'.")

            with tab_data:
                st.subheader("Prediction Results (Top 20 Rows)")
                st.dataframe(results.head(20))
                
                st.subheader("Download Full Results")
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_data = convert_df_to_csv(results)
                st.download_button(
                    label="⬇️ Download Full CSV with Predictions",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            with tab_model:
                st.subheader("About the Model")
                st.info("This tab shows information about the underlying model, not your specific upload. This provides context on how the model was trained.")

                try:
                    fi = model.feature_importances_
                    if len(fi) == len(features):
                        fi_df = pd.DataFrame({"feature": features, "importance": fi})
                        fi_df = fi_df.sort_values("importance", ascending=False).head(20)
                        
                        fig_fi = px.bar(
                            fi_df, 
                            x="importance", 
                            y="feature", 
                            orientation="h", 
                            title="Top 20 Model Feature Importances (XGBoost)"
                        )
                        st.plotly_chart(fig_fi, width='stretch')
                    else:
                        st.info("Feature importances shape doesn't match features; skipping.")
                except Exception as e:
                    st.warning(f"Could not compute feature importances: {e}")

                try:
                    if original_df is not None and "Amount" in original_df.columns and "Class" in original_df.columns:
                        st.subheader("Amount Distribution (Original Training Dataset)")
                        fig_amt = px.histogram(
                            original_df[original_df['Amount'] < 2000], # Cap amount for better visual
                            x="Amount", 
                            color="Class", 
                            nbins=80, 
                            marginal="box", 
                            title="Amount by Class (0=legit, 1=fraud) in Original Data"
                        )
                        st.plotly_chart(fig_amt, width='stretch')
                except Exception as e:
                    st.warning(f"Could not render Amount distribution: {e}")

                # Moved from summary tab
                if "Class" in user_df.columns:
                    st.subheader("Performance on Your Uploaded Data (if 'Class' label provided)")
                    try:
                        # Map true labels if they are 0/1
                        true_labels = user_df["Class"].map({0: "Legit", 1: "Fraud", "0": "Legit", "1": "Fraud"})
                        
                        # Check for NaNs that .map might create if values are already strings
                        if true_labels.isnull().any():
                             true_labels = user_df["Class"].astype(str).map({"0": "Legit", "1": "Fraud", "Legit": "Legit", "Fraud": "Fraud"})

                        cm = confusion_matrix(true_labels, results["Predicted Class"], labels=["Legit", "Fraud"])
                        cm_fig = px.imshow(
                            cm, 
                            text_auto=True, 
                            labels=dict(x="Predicted", y="True", color="Count"), 
                            x=["Legit", "Fraud"], 
                            y=["Legit", "Fraud"],
                            color_continuous_scale="Greens"
                        )
                        cm_fig.update_layout(title="Confusion Matrix")
                        st.plotly_chart(cm_fig, width='stretch')
                        
                        # --- THIS IS THE CHANGED SECTION ---
                        # Display classification report as a formatted DataFrame
                        st.subheader("Classification Report")
                        report_dict = classification_report(true_labels, results["Predicted Class"], labels=["Legit", "Fraud"], output_dict=True)
                        
                        # Extract accuracy separately
                        accuracy = report_dict.pop('accuracy', None)
                        
                        # Convert the rest to DataFrame
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Add accuracy back as a metric
                        if accuracy is not None:
                            st.metric("Accuracy", f"{accuracy:.4f}")
                        
                        # Format support as integer
                        if 'support' in report_df.columns:
                            report_df['support'] = report_df['support'].astype(int)

                        # Display formatted dataframe
                        st.dataframe(report_df.style.format({
                            'precision': '{:.4f}',
                            'recall': '{:.4f}',
                            'f1-score': '{:.4f}',
                        }), width='stretch')
                        # --- END OF CHANGED SECTION ---

                    except Exception as e:
                        st.warning(f"Could not compute confusion matrix. Make sure 'Class' column has 0/1 or 'Legit'/'Fraud' values. Error: {e}")
            
            # --- NEW SHAP EXPLAINABILITY TAB ---
            with tab_explain:
                if explainer is None:
                    st.error("SHAP Explainer could not be loaded. Please ensure all artifacts are present and re-run train_model.py.")
                elif len(X_scaled) == 0:
                    st.info("No data to explain.")
                else:
                    st.subheader("Understand Your Data's Predictions")
                    st.info("""
                    **SHAP (SHapley Additive exPlanations)** shows how each feature contributes to pushing the model's prediction
                    from a "base value" (the average prediction) to the final output (the fraud probability).
                    
                    - **Red bars/features** increase the probability of fraud.
                    - **Blue bars/features** decrease the probability of fraud (i.e., point towards 'Legit').
                    """)

                    # --- REMOVED Local Explanation Section ---
                    max_slider = min(len(X_scaled) - 1, 500)
                    if len(X_scaled) > 500:
                        st.warning(f"Displaying explanations for the first 500 transactions. Your file has {len(X_scaled)} rows.")

                    # --- Global Explanation ---
                    st.header("Global Feature Impact (Summary of this Upload)")
                    st.write("This plot shows the average impact of each feature on *increasing the probability of fraud*.")
                    
                    try:
                        # --- FIX: Re-calculate values if they weren't calculated in the 'local' section ---
                        # (e.g., if slider wasn't used, though it defaults to 0)
                        # This ensures 'shap_explanations' exists.
                        if 'shap_explanations' not in locals():
                             with st.spinner("Calculating SHAP values for your upload... This may take a moment."):
                                X_scaled_subset = X_scaled[:max_slider+1]
                                X_scaled_subset_df = pd.DataFrame(X_scaled_subset, columns=features)
                                shap_explanations = explainer(X_scaled_subset_df)
                        # --- End re-calculation ---

                        st.write("**SHAP Summary Bar Plot:**")
                        st.write("Average absolute SHAP value for each feature.")
                        fig_summary, ax_summary = plt.subplots()
                        # Pass the values from the Explanation object
                        shap.summary_plot(shap_explanations.values, X_scaled_subset_df, feature_names=features, plot_type="bar", show=False, max_display=15)
                        st.pyplot(fig_summary, bbox_inches='tight', clear_figure=True)

                        st.write("**SHAP Beeswarm Plot:**")
                        st.write("Shows feature impact (x-axis) vs. feature value (color). Each dot is a transaction.")
                        fig_beeswarm, ax_beeswarm = plt.subplots()
                        # Pass the full Explanation object
                        shap.summary_plot(shap_explanations, X_scaled_subset_df, feature_names=features, show=False, max_display=15)
                        st.pyplot(fig_beeswarm, bbox_inches='tight', clear_figure=True)
                        # --- END FIX ---
                    
                    except Exception as e:
                        st.error(f"Error generating global SHAP plots: {e}")
            # --- END NEW SHAP TAB ---

st.caption("Credit Card Fraud Detector | Built with XGBoost, SMOTE, and Streamlit")
