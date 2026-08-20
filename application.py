import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e0f14;
        color: white;
    }

    .main {
        padding-top: 2rem;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 700 !important;
        color: #f5f5f5 !important;
    }

    h2, h3 {
        color: #f5f5f5 !important;
    }

    label {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Select boxes */

    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: white !important;
        border-radius: 8px;
    }

    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* Number inputs */

    input {
        background-color: #262730 !important;
        color: white !important;
    }

    div[data-testid="stNumberInput"] > div {
        background-color: #262730 !important;
        border-radius: 8px;
    }

    /* Buttons */

    .stButton > button {
        width: 100%;
        border-radius: 9px;
        height: 3rem;
        font-size: 18px;
        font-weight: 600;
        background-color: transparent;
        color: white;
        border: 1px solid #555;
    }

    .stButton > button:hover {
        border-color: white;
        color: white;
    }

    /* Prediction cards */

    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }

    .churn {
        background-color: #4a2025;
        color: #ff8585;
        border: 1px solid #ff4b4b;
    }

    .no-churn {
        background-color: #173d27;
        color: #72e69a;
        border: 1px solid #21c55d;
    }

    .prediction-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .prediction-probability {
        font-size: 20px;
        font-weight: 500;
        line-height: 1.7;
    }

    .classification {
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE LOCATIONS
# ============================================================

MODEL_FILE = "churn_model.pkl"

DATA_FILES = [
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "telco_churn.csv",
    "Telco-Customer-Churn.csv",
    "customer_churn.csv",
    "data.csv"
]


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():

    for file in DATA_FILES:

        if os.path.exists(file):
            return file

    return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert TotalCharges
    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Convert target to binary classification
    if "Churn" in df.columns:

        df["Churn"] = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .map(
                {
                    "Yes": 1,
                    "No": 0,
                    "1": 1,
                    "0": 0
                }
            )
        )

    # Remove customer ID
    if "customerID" in df.columns:

        df = df.drop(
            columns=["customerID"]
        )

    return df


# ============================================================
# TRAIN CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def train_model():

    dataset_path = find_dataset()

    if dataset_path is None:

        return None, None, None

    # Load dataset
    df = pd.read_csv(
        dataset_path
    )

    # Prepare dataset
    df = prepare_data(
        df
    )

    # Check target
    if "Churn" not in df.columns:

        return None, None, None

    # Remove rows with missing target
    df = df.dropna(
        subset=["Churn"]
    )

    # Features
    X = df.drop(
        columns=["Churn"]
    )

    # Target
    y = df["Churn"].astype(int)

    # ========================================================
    # COLUMN TYPES
    # ========================================================

    numerical_columns = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    # ========================================================
    # NUMERICAL PIPELINE
    # ========================================================

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ========================================================
    # CATEGORICAL PIPELINE
    # ========================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # ========================================================
    # PREPROCESSOR
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # ========================================================
    # CLASSIFICATION MODEL
    # ========================================================

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # TRAIN
    # ========================================================

    pipeline.fit(
        X_train,
        y_train
    )

    # ========================================================
    # TEST
    # ========================================================

    y_pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ],
        output_dict=True
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    try:

        joblib.dump(
            pipeline,
            MODEL_FILE
        )

    except Exception:

        pass

    return (
        pipeline,
        accuracy,
        X.columns.tolist(),
        report,
        matrix
    )


# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================

@st.cache_resource
def load_or_train_model():

    dataset_path = find_dataset()

    # ========================================================
    # LOAD EXISTING MODEL
    # ========================================================

    if os.path.exists(MODEL_FILE):

        try:

            model = joblib.load(
                MODEL_FILE
            )

            columns = None

            if dataset_path is not None:

                df = pd.read_csv(
                    dataset_path
                )

                df = prepare_data(
                    df
                )

                if "Churn" in df.columns:

                    columns = df.drop(
                        columns=["Churn"]
                    ).columns.tolist()

            return (
                model,
                None,
                columns,
                None,
                None
            )

        except Exception:

            pass

    # ========================================================
    # TRAIN NEW MODEL
    # ========================================================

    return train_model()


model, accuracy, model_columns, report, matrix = (
    load_or_train_model()
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📞 Customer Churn Prediction"
)

st.write(
    "Enter the customer details below."
)


# ============================================================
# INPUT FORM
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    customer_id = st.number_input(
        "Customer ID",
        min_value=1,
        value=225,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ],
        index=0
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [
            0,
            1
        ],
        index=1
    )

    partner = st.selectbox(
        "Partner",
        [
            "Yes",
            "No"
        ],
        index=0
    )

    dependents = st.selectbox(
        "Dependents",
        [
            "Yes",
            "No"
        ],
        index=0
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        [
            "Yes",
            "No"
        ],
        index=0
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No phone service",
            "No",
            "Yes"
        ],
        index=1
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ],
        index=1
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No internet service",
            "No",
            "Yes"
        ],
        index=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No internet service",
            "No",
            "Yes"
        ],
        index=1
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No internet service",
            "No",
            "Yes"
        ],
        index=1
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No internet service",
            "No",
            "Yes"
        ],
        index=1
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No"
        ],
        index=0
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No"
        ],
        index=0
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ],
        index=0
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ],
        index=0
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        index=0
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.00,
        step=1.0,
        format="%.2f"
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.00,
        step=10.0,
        format="%.2f"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Churn",
    type="primary"
)


# ============================================================
# CLASSIFICATION PREDICTION
# ============================================================

if predict_button:

    if model is None:

        st.error(
            "Classification model could not be loaded. "
            "Please place the Telco Customer Churn CSV "
            "in the same folder as app.py."
        )

    else:

        # ====================================================
        # CREATE CUSTOMER DATAFRAME
        # ====================================================

        customer_data = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges]
            }
        )

        # ====================================================
        # MATCH TRAINING COLUMNS
        # ====================================================

        if model_columns is not None:

            for column in model_columns:

                if column not in customer_data.columns:

                    customer_data[column] = np.nan

            customer_data = customer_data[
                model_columns
            ]

        try:

            # =================================================
            # CLASSIFICATION
            # =================================================

            prediction = model.predict(
                customer_data
            )[0]

            # =================================================
            # PROBABILITY
            # =================================================

            probability = model.predict_proba(
                customer_data
            )[0][1]

            churn_probability = (
                probability * 100
            )

            stay_probability = (
                100 - churn_probability
            )

            # =================================================
            # 0-9 CHURN SCORE
            # =================================================

            churn_score = round(
                probability * 9
            )

            # Keep score between 0 and 9
            churn_score = max(
                0,
                min(
                    9,
                    churn_score
                )
            )

            # =================================================
            # CLASS 1 = CHURN
            # =================================================

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="prediction-card churn">

                        <div class="prediction-title">
                            ⚠️ Customer is likely to CHURN
                        </div>

                        <div class="prediction-probability">

                            Classification:
                            <strong>CHURN (1)</strong>

                            <br>

                            Churn Probability:
                            <strong>
                                {churn_probability:.2f}%
                            </strong>

                            <br>

                            Churn Score:
                            <strong>
                                {churn_score}/9
                            </strong>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # CLASS 0 = NO CHURN
            # =================================================

            else:

                st.markdown(
                    f"""
                    <div class="prediction-card no-churn">

                        <div class="prediction-title">
                            ✅ Customer is unlikely to CHURN
                        </div>

                        <div class="prediction-probability">

                            Classification:
                            <strong>NO CHURN (0)</strong>

                            <br>

                            Churn Probability:
                            <strong>
                                {churn_probability:.2f}%
                            </strong>

                            <br>

                            Churn Score:
                            <strong>
                                {churn_score}/9
                            </strong>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # PROBABILITY
            # =================================================

            st.subheader(
                "Classification Probability"
            )

            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:

                st.metric(
                    "No Churn (Class 0)",
                    f"{stay_probability:.2f}%"
                )

            with probability_col2:

                st.metric(
                    "Churn (Class 1)",
                    f"{churn_probability:.2f}%"
                )

            st.progress(
                int(churn_probability)
            )

            # =================================================
            # CUSTOMER DETAILS
            # =================================================

            with st.expander(
                "View Customer Details"
            ):

                summary = pd.DataFrame(
                    {
                        "Feature": [
                            "Customer ID",
                            "Gender",
                            "Senior Citizen",
                            "Partner",
                            "Dependents",
                            "Tenure (Months)",
                            "Phone Service",
                            "Multiple Lines",
                            "Internet Service",
                            "Online Security",
                            "Online Backup",
                            "Device Protection",
                            "Tech Support",
                            "Streaming TV",
                            "Streaming Movies",
                            "Contract",
                            "Paperless Billing",
                            "Payment Method",
                            "Monthly Charges",
                            "Total Charges"
                        ],
                        "Value": [
                            customer_id,
                            gender,
                            senior_citizen,
                            partner,
                            dependents,
                            tenure,
                            phone_service,
                            multiple_lines,
                            internet_service,
                            online_security,
                            online_backup,
                            device_protection,
                            tech_support,
                            streaming_tv,
                            streaming_movies,
                            contract,
                            paperless_billing,
                            payment_method,
                            f"${monthly_charges:.2f}",
                            f"${total_charges:.2f}"
                        ]
                    }
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                f"Classification failed: {str(e)}"
            )


# ============================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    if model is not None:

        st.success(
            "Classification Model Loaded"
        )

        st.write(
            "**Algorithm:** Logistic Regression"
        )

        st.write(
            "**Problem:** Binary Classification"
        )

        st.write(
            "**Class 0:** No Churn"
        )

        st.write(
            "**Class 1:** Churn"
        )

        st.write(
            "**Encoding:** One-Hot Encoding"
        )

        st.write(
            "**Scaling:** StandardScaler"
        )

        if accuracy is not None:

            st.metric(
                "Model Accuracy",
                f"{accuracy * 100:.2f}%"
            )

    else:

        st.error(
            "Model not available"
        )

    st.divider()

    st.caption(
        "Customer Churn Prediction"
    )
