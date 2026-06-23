import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)



# ============================================================
# STREAMLIT APP — CHURN PREDICTION WITH SHAP EXPLANATION
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)


# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS (cached so it only loads once, not on every click)
# ─────────────────────────────────────────────


@st.cache_resource
def load_artifacts():
    """Load the trained model bundle and SHAP explainers from disk."""
    # NOTE: these files must sit in the SAME folder you run
    # `streamlit run app.py` from. If you get FileNotFoundError,
    # that's almost always the cause — check your terminal's
    # current directory with `pwd` (Mac/Linux) or `cd` (Windows).
    bundle = joblib.load('churn_model_bundle.pkl')
    shap_artifacts = joblib.load('shap_artifacts.pkl')
    return bundle, shap_artifacts

bundle, shap_artifacts = load_artifacts()

# Unpack everything we need from the saved bundle
fitted_pipelines    = bundle['fitted_pipelines']      # dict of {name: fitted sklearn Pipeline}
meta_model           = bundle['meta_model']            # fitted GradientBoosting meta model
threshold             = bundle['threshold']             # best F1 threshold found during training
numcols               = bundle['numcols']
binary_columns        = bundle['binary_columns']
non_binary_columns    = bundle['non_binary_columns']

explainer_base   = shap_artifacts['explainer_base']    # SHAP explainer for the RF base model
feature_names    = shap_artifacts['feature_names']     # readable column names after preprocessing


# ─────────────────────────────────────────────
# APP TITLE & DESCRIPTION
# ─────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.write(
    "Fill in a customer's details below to predict their churn risk "
    "and see exactly which factors are driving that prediction."
)

st.divider()


# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
# st.form groups all inputs together so the app only re-runs
# ONCE when the user clicks "Predict" — not after every single
# field change, which would feel sluggish.

with st.form("customer_form"):
    st.subheader("Customer Details")

    # Using columns to arrange the form neatly side by side
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

    # The submit button — clicking this triggers everything below
    submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)


# ─────────────────────────────────────────────
# WHEN THE USER SUBMITS THE FORM
# ─────────────────────────────────────────────
if submitted:

    # ───── Step 1: Build a single-row DataFrame from form inputs ─────
    raw_input = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': '1' if senior_citizen == 'Yes' else '0',  # match training format (string)
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }])

    # ───── Step 2: Recreate the SAME engineered features used in training ─────
    def tenure_bucket(t):
        if t <= 12: return 'new'
        elif t < 36: return 'growing'
        elif t < 60: return 'loyal'
        else: return 'champion'

    raw_input['tenure_group'] = raw_input['tenure'].apply(tenure_bucket)
    raw_input['charge_gap'] = (raw_input['MonthlyCharges'] * raw_input['tenure']) - raw_input['TotalCharges']
    raw_input['no_support'] = (
        (raw_input['OnlineSecurity'] == 'No') & (raw_input['TechSupport'] == 'No') &
        (raw_input['DeviceProtection'] == 'No') & (raw_input['OnlineBackup'] == 'No')
    ).astype(int).astype(str)
    raw_input['senior_paperless'] = (
        (raw_input['SeniorCitizen'] == '1') & (raw_input['PaperlessBilling'] == 'Yes')
    ).astype(int).astype(str)
    raw_input['charge_per_month'] = raw_input['TotalCharges'] / raw_input['tenure'].clip(lower=1)

    # ───── Step 2b: Map binary text columns to 0/1 — THIS WAS MISSING ─────
    # During training, every binary categorical column (gender, Partner,
    # Dependents, PhoneService, SeniorCitizen, OnlineSecurity, etc.) was
    # converted from text ('Yes'/'No'/'Male'/'Female'/'0'/'1') into actual
    # integers 0/1 BEFORE it ever reached the pipeline. The pipeline's
    # ColumnTransformer treats binary columns as 'passthrough' — meaning
    # it does ZERO conversion itself. It assumes the values are already
    # numeric by the time they arrive.
    #
    # Streamlit was sending raw text straight into predict_proba(), which
    # is exactly why LogisticRegression crashed inside numpy.asarray() —
    # it tried to convert strings like 'Male' or 'No' into floats and failed.
    #
    # FIX: replicate the EXACT same mapping used in training, applied to
    # the SAME set of binary columns, before calling predict_proba().

    binary_columns = bundle['binary_columns']   # loaded from the saved bundle — must match training exactly

    def mapped_binary(df, col):
        """Identical to the function used during training — must stay in sync."""
        mapping = {'No': 0, 'Yes': 1, 'Male': 1, 'Female': 0, '0': 0, '1': 1}
        df[col] = df[col].map(mapping)
        return df

    for col in binary_columns:
        raw_input = mapped_binary(raw_input, col)

    # ───── Step 2c: Guard against silent NaN from unmapped categories ─────
    # If a form value isn't in the mapping dict (e.g. "No internet service"
    # selected for a column your TRAINING DATA never saw as binary), .map()
    # silently produces NaN instead of raising an error. NaN reaches
    # numpy.asarray() inside the model and crashes with the exact same
    # error you saw — but with no indication of WHICH column caused it.
    # This check fails loudly and tells you exactly which column is wrong.
    nan_cols = raw_input[binary_columns].columns[raw_input[binary_columns].isna().any()].tolist()
    if nan_cols:
        st.error(
            f"⚠️ Input error: these fields have a value that wasn't seen "
            f"during training as binary: {nan_cols}. "
            f"Check the form options for these fields."
        )
        st.stop()   # halts execution here — prevents the crash below

    # Sanity check (optional but recommended while debugging):
    # st.write(raw_input.dtypes)  ← uncomment to inspect dtypes if errors persist

    # ───── Step 3: Run through EACH base model pipeline ─────
    base_probs = {}
    for name, pipe in fitted_pipelines.items():
        prob = pipe.predict_proba(raw_input)[:, 1][0]   # probability of churn (class 1)
        base_probs[f'{name}_prob'] = prob

    # ───── Step 4: Feed those base probabilities into the meta model ─────
    stack_input = pd.DataFrame([base_probs])
    final_prob = meta_model.predict_proba(stack_input)[:, 1][0]
    final_pred = int(final_prob > threshold)

    # ───── Step 5: Display the prediction ─────
    st.divider()
    st.subheader("Prediction Result")

    risk_pct = final_prob * 100

    if final_pred == 1:
        st.error(f"⚠️ **High Churn Risk** — {risk_pct:.1f}% probability")
    else:
        st.success(f"✅ **Low Churn Risk** — {risk_pct:.1f}% probability")

    st.progress(min(final_prob, 1.0))
    st.caption(f"Decision threshold used: {threshold:.2f} (probabilities above this are flagged as churn risk)")

    # ───── Step 6: SHAP explanation for THIS customer ─────
    st.divider()
    st.subheader("Why this prediction? (SHAP Explanation)")
    st.write(
        "The chart below shows which factors pushed this customer's "
        "risk score up (red) or down (blue), using our best-performing "
        "base model (Random Forest) for interpretability."
    )

    # Transform this customer's data through the SAME preprocessing
    # pipeline used by the Random Forest model, so SHAP sees exactly
    # what the model sees.
    rf_pipeline = fitted_pipelines['rf']
    preprocessor = rf_pipeline.named_steps['preprocessing']
    x_transformed = preprocessor.transform(raw_input)
    x_transformed_df = pd.DataFrame(x_transformed, columns=feature_names)

    # Compute SHAP values for this single customer
    shap_values = explainer_base.shap_values(x_transformed_df)

    # Handle SHAP output shape — varies by SHAP version / model type
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
        expected_val = explainer_base.expected_value[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_row = shap_values[0, :, 1]
        expected_val = explainer_base.expected_value[1]
    else:
        shap_row = shap_values[0]
        expected_val = explainer_base.expected_value

    # Draw the waterfall plot and display it inside the Streamlit app
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_row,
            base_values=expected_val,
            data=x_transformed_df.iloc[0].values,
            feature_names=feature_names
        ),
        show=False,
        max_display=10
    )
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

    st.caption(
        "Red bars push the prediction toward churn; blue bars push it "
        "toward staying. Bars are ordered by impact size."
    )
