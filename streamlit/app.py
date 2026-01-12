import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Digital Marketing Conversion Predictor",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("digital_marketing_campaign_dataset.csv")

df = load_data()

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("Marketing Analytics App")
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Preview", "EDA Dashboard", "Model Training", "Conversion Prediction"]
)

# -----------------------------------
# Dataset Preview
# -----------------------------------
if page == "Dataset Preview":
    st.title("📁 Dataset Overview")

    st.metric("Total Records", df.shape[0])
    st.metric("Total Features", df.shape[1])

    st.dataframe(df.head(100), use_container_width=True)

# -----------------------------------
# EDA Dashboard
# -----------------------------------
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Conversion Rate", f"{df['Conversion'].mean()*100:.2f}%")
    col2.metric("Avg Ad Spend", f"₹ {df['AdSpend'].mean():,.0f}")
    col3.metric("Avg Time on Site", f"{df['TimeOnSite'].mean():.2f} mins")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conversion by Campaign Type")
        fig, ax = plt.subplots()
        sns.barplot(x="CampaignType", y="Conversion", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Conversion by Channel")
        fig, ax = plt.subplots()
        sns.barplot(x="CampaignChannel", y="Conversion", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Ad Spend vs Conversion")
    fig, ax = plt.subplots()
    sns.boxplot(x="Conversion", y="AdSpend", data=df, ax=ax)
    st.pyplot(fig)

# -----------------------------------
# Model Training
# -----------------------------------
elif page == "Model Training":
    st.title("🤖 Train Conversion Prediction Model")

    target = "Conversion"

    X = df.drop(columns=[target, "CustomerID"])
    y = df[target]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    test_size = st.slider("Test Size", 0.2, 0.4, 0.25)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)

        st.success(f"Model Trained Successfully 🎉")
        st.metric("Accuracy", f"{acc:.3f}")

        st.text("Classification Report")
        st.code(classification_report(y_test, preds))

        joblib.dump(pipeline, "conversion_model.pkl")
        st.info("Model saved as conversion_model.pkl")

# -----------------------------------
# Prediction Page
# -----------------------------------
elif page == "Conversion Prediction":
    st.title("🔮 Predict Customer Conversion")

    model = joblib.load("conversion_model.pkl")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 80)
            income = st.number_input("Income", 10000, 200000)
            adspend = st.number_input("Ad Spend")

        with col2:
            ctr = st.number_input("Click Through Rate")
            pages = st.number_input("Pages Per Visit")
            time = st.number_input("Time On Site")

        with col3:
            gender = st.selectbox("Gender", df["Gender"].unique())
            channel = st.selectbox("Campaign Channel", df["CampaignChannel"].unique())
            ctype = st.selectbox("Campaign Type", df["CampaignType"].unique())

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "AdSpend": adspend,
            "ClickThroughRate": ctr,
            "PagesPerVisit": pages,
            "TimeOnSite": time,
            "Gender": gender,
            "CampaignChannel": channel,
            "CampaignType": ctype,
            "ConversionRate": 0,
            "WebsiteVisits": 0,
            "SocialShares": 0,
            "EmailOpens": 0,
            "EmailClicks": 0,
            "PreviousPurchases": 0,
            "LoyaltyPoints": 0,
            "AdvertisingPlatform": df["AdvertisingPlatform"].mode()[0],
            "AdvertisingTool": df["AdvertisingTool"].mode()[0]
        }])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"Likely to Convert (Probability: {prob:.2%})")
        else:
            st.error(f"Unlikely to Convert (Probability: {prob:.2%})")
