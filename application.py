import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    try:
        data = pd.read_csv("cleaned_data.csv")
    except FileNotFoundError:
        st.error("❌ cleaned_data.csv file not found.")
        st.stop()

    # Remove unwanted index column
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    return data


df = load_data()


# ============================================================
# CHECK DATASET
# ============================================================

if df.empty:
    st.error("❌ The dataset is empty.")
    st.stop()


if "Churn" not in df.columns:
    st.error(
        "❌ 'Churn' column is missing from cleaned_data.csv."
    )

    st.write("Available columns:")
    st.write(df.columns.tolist())

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# Fill numeric missing values
numeric_cols = df.select_dtypes(
    include=["number"]
).columns

for column in numeric_cols:

    if df[column].isna().any():

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(
            median_value
        )


# Fill categorical missing values
categorical_cols = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_cols:

    df[column] = df[column].fillna("Unknown")


# ============================================================
# PREPARE CHURN TARGET
# ============================================================

def convert_churn(value):

    # Already numeric
    if isinstance(value, (int, float)):

        if value == 1:
            return 1

        if value == 0:
            return 0

    value = str(value).strip().lower()

    if value in [
        "yes",
        "y",
        "1",
        "true",
        "churn",
        "churned"
    ]:
        return 1

    if value in [
        "no",
        "n",
        "0",
        "false",
        "not churn",
        "not churned"
    ]:
        return 0

    return None


y = df["Churn"].apply(convert_churn)


# Remove invalid target rows
valid_rows = y.notna()

df = df.loc[valid_rows].copy()

y = y.loc[valid_rows].astype(int)


# ============================================================
# CHECK TARGET
# ============================================================

if len(df) < 2:

    st.error(
        "❌ Not enough valid records to train the model."
    )

    st.stop()


if y.nunique() < 2:

    st.error(
        "❌ Churn column must contain both "
        "Churn and Not Churn records."
    )

    st.stop()


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df.drop(
    columns=["Churn"]
)


# Customer ID is only an identifier
# and should not be used for prediction.

if "customerID" in X.columns:

    X = X.drop(
        columns=["customerID"]
    )


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            categorical_columns
        ),

        (
            "numeric",

            "passthrough",

            numeric_columns
        )
    ]
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

classifier = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


# ============================================================
# PIPELINE
# ============================================================

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


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X, y):

    class_counts = y.value_counts()

    # If dataset is very small
    if len(X) < 5:

        pipeline.fit(
            X,
            y
        )

        return pipeline, None


    # Stratified split only when
    # every class has at least 2 records.

    if class_counts.min() >= 2:

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42
        )


    # Train model
    pipeline.fit(
        X_train,
        y_train
    )


    # Test model
    predictions = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return pipeline, accuracy


# ============================================================
# TRAIN
# ============================================================

pipeline, accuracy = train_model(
    X,
    y
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📞 Customer Churn Prediction"
)

st.write(
    "Enter the customer details below "
    "to predict customer churn."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if accuracy is not None:

    st.info(
        f"🤖 Model Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

else:

    st.info(
        "🤖 Model trained successfully."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

with col1:

    st.subheader(
        "👤 Customer Information"
    )


    customerID = st.text_input(
        "Customer ID",
        value="CUST001"
    )


    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [
            0,
            1
        ]
    )


    Partner = st.selectbox(
        "Partner",
        [
            "Yes",
            "No"
        ]
    )


    Dependents = st.selectbox(
        "Dependents",
        [
            "Yes",
            "No"
        ]
    )


    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


# ============================================================
# SERVICES
# ============================================================

with col2:

    st.subheader(
        "📱 Services"
    )


    PhoneService = st.selectbox(
        "Phone Service",
        [
            "Yes",
            "No"
        ]
    )


    MultipleLines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )


    InternetService = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


    OnlineSecurity = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    OnlineBackup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    DeviceProtection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    TechSupport = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# ============================================================
# SUBSCRIPTION INFORMATION
# ============================================================

with col3:

    st.subheader(
        "📺 Subscription"
    )


    StreamingTV = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    StreamingMovies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )


    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()


if st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame([

        {

            "customerID": customerID,

            "gender": gender,

            "SeniorCitizen": SeniorCitizen,

            "Partner": Partner,

            "Dependents": Dependents,

            "tenure": tenure,

            "PhoneService": PhoneService,

            "MultipleLines": MultipleLines,

            "InternetService": InternetService,

            "OnlineSecurity": OnlineSecurity,

            "OnlineBackup": OnlineBackup,

            "DeviceProtection": DeviceProtection,

            "TechSupport": TechSupport,

            "StreamingTV": StreamingTV,

            "StreamingMovies": StreamingMovies,

            "Contract": Contract,

            "PaperlessBilling": PaperlessBilling,

            "PaymentMethod": PaymentMethod,

            "MonthlyCharges": MonthlyCharges,

            "TotalCharges": TotalCharges

        }

    ])


    # --------------------------------------------------------
    # Remove customer ID
    # --------------------------------------------------------

    if "customerID" in input_data.columns:

        input_data = input_data.drop(
            columns=["customerID"]
        )


    # --------------------------------------------------------
    # Match training columns
    # --------------------------------------------------------

    # Add missing training columns
    for column in X.columns:

        if column not in input_data.columns:

            if column in numeric_columns:

                input_data[column] = 0

            else:

                input_data[column] = "Unknown"


    # Keep exactly the same columns
    # and same order as training data.

    input_data = input_data[
        X.columns
    ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = pipeline.predict(
            input_data
        )[0]


        # Probability
        probabilities = pipeline.predict_proba(
            input_data
        )[0]


        churn_probability = probabilities[1]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Result"
        )


        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to Churn."
            )

        else:

            st.success(
                "✅ Customer is NOT likely to Churn."
            )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )


        # Progress bar
        st.progress(
            float(churn_probability)
        )


        # ----------------------------------------------------
        # Customer Summary
        # ----------------------------------------------------

        st.subheader(
            "📋 Customer Summary"
        )


        summary_col1, summary_col2, summary_col3 = st.columns(3)


        with summary_col1:

            st.write(
                f"**Customer ID:** {customerID}"
            )

            st.write(
                f"**Gender:** {gender}"
            )

            st.write(
                f"**Senior Citizen:** {SeniorCitizen}"
            )


        with summary_col2:

            st.write(
                f"**Tenure:** {tenure} months"
            )

            st.write(
                f"**Contract:** {Contract}"
            )

            st.write(
                f"**Internet Service:** {InternetService}"
            )


        with summary_col3:

            st.write(
                f"**Monthly Charges:** "
                f"${MonthlyCharges:.2f}"
            )

            st.write(
                f"**Total Charges:** "
                f"${TotalCharges:.2f}"
            )

            st.write(
                f"**Payment Method:** "
                f"{PaymentMethod}"
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander(
    "📁 Dataset Information"
):

    st.write(
        f"Total records: {len(df)}"
    )

    st.write(
        f"Total features: {len(X.columns)}"
    )

    st.write(
        "Features used by model:"
    )

    st.write(
        X.columns.tolist()
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📞 Customer Churn Prediction System "
    "| Built with Streamlit & Machine Learning"
)
