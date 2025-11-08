"""
Credit Card Fraud Detection Model Training Pipeline
- Uses ColumnTransformer + RobustScaler ONLY on Time and Amount
- SMOTE applied only on training set
- Saves: model, scaler, feature names, SHAP background data

Note: For quick runs in constrained environments this script samples the dataset.
Set SAMPLE_N=None to use full dataset.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib, json, numpy as np
import shap  # <-- Import SHAP

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"
SAMPLE_N = 50000  # set to None to use full dataset

# --- Data Cleaning Helpers ---
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
# --- End of helpers ---

def train_and_save(random_state: int = 42):
    print("Loading dataset...")
    if not DATA_PATH.exists():
        print(f"❌ Error: {DATA_PATH} not found.")
        print("Please download 'creditcard.csv' from Kaggle and place it in the same directory.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    print("Cleaning raw dataset...")
    feature_cols = [col for col in df.columns if col != 'Class']
    df = clean_numeric_df(df, cols=feature_cols)
    
    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        print(f"Sampling {SAMPLE_N} total rows stratified by target to speed up training...")
        # Use train_test_split to sample stratified
        df, _ = train_test_split(
            df, 
            train_size=SAMPLE_N, 
            random_state=random_state, 
            stratify=df['Class']
        )
        df = df.reset_index(drop=True)

    target = "Class"
    X = df.drop(columns=[target])
    y = df[target]

    feature_names = list(X.columns)
    # Ensure 'Time' and 'Amount' are present, even if V features are dynamic
    if 'Time' not in feature_names or 'Amount' not in feature_names:
        print("Error: 'Time' or 'Amount' columns not found in dataset.")
        return
        
    v_features = [col for col in feature_names if col.startswith('V')]
    
    # Define the transformer:
    # 1. Scale 'Time'
    # 2. Pass through all 'V' features
    # 3. Scale 'Amount'
    ct = ColumnTransformer([
        ("scale_time", RobustScaler(), ['Time']),
        ("pass_V", "passthrough", v_features),
        ("scale_amount", RobustScaler(), ['Amount'])
    ], remainder="drop", verbose_feature_names_out=False)

    # Re-order feature_names to match ColumnTransformer output for the model
    # This is critical for feature importance mapping
    feature_names_transformed = ['Time'] + v_features + ['Amount']
    
    # Check if all original features are accounted for
    if set(feature_names_transformed) != set(feature_names):
        print("Feature name mismatch during transform setup. Aborting.")
        print(f"Expected: {set(feature_names)}")
        print(f"Got: {set(feature_names_transformed)}")
        return
    
    # We pass the original X to train_test_split, ensuring correct column order
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_names_transformed], # Ensure X is in the correct order
        y, 
        test_size=0.2, 
        random_state=random_state, 
        stratify=y
    )

    print("Fitting scaler...")
    # Fit transformer on the training data
    ct.fit(X_train)
    X_train_scaled = ct.transform(X_train)
    X_test_scaled = ct.transform(X_test)

    print("🔁 Applying SMOTE on training set...")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    
    n_minority = y_train.sum()
    if n_minority == 0:
        print("No minority samples found in training set. Skipping SMOTE.")
        X_train_res, y_train_res = X_train_scaled, y_train
    else:
        # Adjust k_neighbors if it's larger than the number of minority samples
        if n_minority <= sm.k_neighbors:
            print(f"Warning: Minority class ({n_minority}) is smaller than/equal to k_neighbors ({sm.k_neighbors}).")
            sm.k_neighbors = max(1, n_minority - 1) # k_neighbors must be at least 1
            print(f"Adjusting k_neighbors to {sm.k_neighbors}.")

        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    print("Training XGBoost model...")
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="auc",
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train_res, y_train_res)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    print(classification_report(y_test, y_pred, digits=4))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    print("Saving artifacts...")
    joblib.dump(model, BASE_DIR / "fraud_model.pkl")
    joblib.dump(ct, BASE_DIR / "scaler.pkl")
    # Save the *transformed* feature names order, which app.py expects
    with open(BASE_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names_transformed, f)

    # --- SHAP SECTION ---
    print("Creating SHAP background dataset...")
    # Sample 100 data points from the scaled training set (pre-SMOTE) for SHAP background
    # We use X_train_scaled as it represents the real data distribution
    if len(X_train_scaled) > 100:
        shap_background = shap.sample(X_train_scaled, 100)
    else:
        shap_background = X_train_scaled
    
    # Save the background dataset
    np.save(BASE_DIR / "shap_background.npy", shap_background)
    # --- END SHAP SECTION ---

    print("\nTraining completed successfully!\n")

if __name__ == "__main__":
    train_and_save()
