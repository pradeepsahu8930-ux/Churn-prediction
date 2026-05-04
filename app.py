import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="AI Churn Dashboard", layout="wide")

# Load model
model = joblib.load("churn_model.pkl")

# Sidebar navigation
menu = st.sidebar.radio("📌 Navigation", ["Home", "Prediction", "Insights"])

# -------------------------
# 🏠 HOME PAGE
# -------------------------
if menu == "Home":

    st.title("🚀 AI Customer Churn Intelligence System")
    st.markdown("### 🧠 Model Features Used")
    st.write("""
    - Tenure (Months)  
    - Monthly Charges  
    - Contract Type  
    - Internet Service  
    """)

    st.markdown("### ⚙ How System Works")
    st.write("""
    1. Data is collected from Netflix,Prime Video,Hotstar  
    2. Important features are selected  
    3. Machine learning model is trained  
    4. Model predicts churn probability  
    """)


    # 🔥 Dropdown inside Home page
    platform = st.selectbox(
        "🎬 Select Platform Dataset",
        ["Netflix", "Prime Video", "Hotstar"]
    )

    # Load dataset based on selection
    if platform == "Netflix":
        df = pd.read_csv("telco_churn.csv")
    elif platform == "Prime Video":
        df = pd.read_csv("prime.csv")
    else:
        df = pd.read_csv("hotstar.csv")

    # Fix churn label
    if df["Churn"].dtype == "object":
        df["Churn_Label"] = df["Churn"].map({"No": "Stayed", "Yes": "Churned"})
    else:
        df["Churn_Label"] = df["Churn"].map({0: "Stayed", 1: "Churned"})

    # Dataset preview
    st.subheader(f"📂 {platform} Dataset Overview")
    st.dataframe(df.head())

    # Columns
    st.subheader("📊 Columns")
    st.write(list(df.columns))

    # Metrics
    total = len(df)
    churned = int((df["Churn_Label"] == "Churned").sum())
    retained = int((df["Churn_Label"] == "Stayed").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", total)
    c2.metric("Churned", churned)
    c3.metric("Retained", retained)
    

# -------------------------
# 🤖 PREDICTION PAGE
# -------------------------
elif menu == "Prediction":

    st.title("🤖 Churn Prediction Engine")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("📅 Tenure (Months)", 1, 72)
        monthly = st.slider("💰 Monthly Charges ($)", 10, 120)

        contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("🔍 Analyze Customer"):

        input_df = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly],
            "Contract": [contract],
            "InternetService": [internet]
        })

        input_df = pd.get_dummies(input_df)

        model_columns = model.feature_names_in_
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("📊 Prediction Result")
        st.metric("Churn Probability", f"{proba*100:.2f}%")

        if proba > 0.7:
            st.error("⚠ High Risk Customer")
        elif proba > 0.4:
            st.warning("⚠ Medium Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

# -------------------------
# 📊 INSIGHTS PAGE
# -------------------------
elif menu == "Insights":

    st.title("📊 Customer Insights Dashboard")

    # Default dataset (you can also add dropdown here later)
   # 🔥 Dropdown inside Home page
    platform = st.selectbox(
        "🎬 Select Platform Dataset",
        ["Netflix", "Prime Video", "Hotstar"]
    )

    # Load dataset based on selection
    if platform == "Netflix":
        df = pd.read_csv("telco_churn.csv")
    elif platform == "Prime Video":
        df = pd.read_csv("prime.csv")
    else:
        df = pd.read_csv("hotstar.csv")

    if df["Churn"].dtype == "object":
        df["Churn_Label"] = df["Churn"].map({"No": "Stayed", "Yes": "Churned"})
    else:
        df["Churn_Label"] = df["Churn"].map({0: "Stayed", 1: "Churned"})

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="Churn_Label",
            color="Churn_Label",
            title="Churn Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.box(
            df,
            x="Churn_Label",
            y="MonthlyCharges",
            color="Churn_Label",
            title="Monthly Charges vs Churn",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🧠 Key Insights")
    st.write("""
    - Low tenure customers churn more  
    - High monthly charges increase churn  
    - Contract type impacts churn  
    """)