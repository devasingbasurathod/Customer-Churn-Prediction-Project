import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e0f14;
        color: white;
    }

    label {
        color: white !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: white !important;
        border-radius: 8px;
    }

    div[data-baseweb="select"] span {
        color: white !important;
    }

    input {
        background-color: #262730 !important;
        color: white !important;
    }

    div[data-testid="stNumberInput"] > div {
        background-color: #262730 !important;
        border-radius: 8px;
    }

    .stButton button {
        background-color: transparent;
        color: white;
        border: 1px solid #555;
        border-radius: 9px;
    }

    .stButton button:hover {
        border-color: white;
        color: white;
    }

    .success {
        background-color: #173d27;
        color: #72e69a;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }

    .danger {
        background-color: #4a2025;
        color: #ff8585;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Input fields
# -------------------------

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"],
    index=0
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"],
    index=0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"],
    index=0
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
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


# -------------------------
# Churn prediction
# -------------------------

if st.button("🔍 Predict Churn"):

    score = 0

    # Contract
    if contract == "Month-to-month":
        score += 2
    elif contract == "One year":
        score += 1

    # Payment method
    if payment_method == "Electronic check":
        score += 1

    # Paperless billing
    if paperless_billing == "Yes":
        score += 1

    # Monthly charges
    if monthly_charges >= 70:
        score += 1

    # Streaming services
    if streaming_tv == "No":
        score += 1

    if streaming_movies == "No":
        score += 1

    # Total charges
    if total_charges < 1000:
        score += 1

    # Final prediction
    if score >= 4:
        st.markdown(
            """
            <div class="danger">
                ⚠️ Customer is likely to churn
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success">
                ✅ Customer is unlikely to churn
            </div>
            """,
            unsafe_allow_html=True
        )
