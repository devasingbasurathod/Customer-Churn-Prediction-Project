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
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")

    # Remove unnecessary index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


df = load_data()


# ============================================================
# PREPARE DATA
# ============================================================

# Convert TotalCharges to numeric if it exists
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Fill missing numeric values
for column in df.select_dtypes(include=["number"]).columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values
for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].fillna("Unknown")


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if "Churn" not in df.columns:
    st.error(
        "❌ The column 'Churn' was not found in cleaned_data.csv. "
        "Please check your CSV file."
    )
    st.stop()


# ============================================================
# PREPARE TARGET
# ============================================================

y = df["Churn"].copy()

# Convert Yes/No target into 1/0
if y.dtype == "object":
    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
    )

# Remove rows where target could not be converted
valid_rows = y.notna()

df = df.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df.drop(columns=["Churn"])

# Customer ID is an identifier, not a useful predictive feature
if "customerID" in X.columns:
    X = X.drop(columns=["customerID"])


# ============================================================
# IDENTIFY COLUMNS
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
# MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return pipeline, accuracy


pipeline, accuracy = train_model(X, y)


# ============================================================
# HEADER
# ============================================================

st.title("📞 Customer Churn Prediction")
st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to churn."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.info(
    f"🤖 Model Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# INPUT FIELDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("👤 Customer Information")

    customerID = st.text_input(
        "Customer ID",
        value="CUST001"
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    st.subheader("📱 Services")

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    st.subheader("📺 Subscription")

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
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
        ["Yes", "No"]
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
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
):

    input_data = pd.DataFrame([{

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

    }])


    # Remove Customer ID because it wasn't used for training
    if "customerID" in input_data.columns:
        input_data = input_data.drop(
            columns=["customerID"]
        )


    # Make prediction
    prediction = pipeline.predict(input_data)[0]


    # Get probability
    probability = pipeline.predict_proba(
        input_data
    )[0][1]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to Churn."
        )

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    else:

        st.success(
            "✅ Customer is NOT likely to Churn."
        )

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📞 Customer Churn Prediction System | "
    "Built with Streamlit & Machine Learning"
)
