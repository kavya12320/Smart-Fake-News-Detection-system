
import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #F1F5F9;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #172554;
        text-align: center;
    }

    .sub-title {
        font-size: 16px;
        color: #64748B;
        text-align: center;
        margin-bottom: 25px;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #2563EB);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h2 {
        color: white;
        margin-bottom: 8px;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #DBEAFE;
        text-align: center;
        min-height: 115px;
    }

    .metric-number {
        font-size: 28px;
        font-weight: bold;
        color: #1D4ED8;
    }

    .metric-label {
        font-size: 14px;
        color: #64748B;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
    }

    .real-box {
        background: #DCFCE7;
        border: 2px solid #16A34A;
        color: #14532D;
    }

    .fake-box {
        background: #FEE2E2;
        border: 2px solid #DC2626;
        color: #7F1D1D;
    }

    .footer {
        text-align: center;
        color: #64748B;
        padding: 25px;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #DBEAFE;
    }

    /* ======================================
       MAIN PAGE TEXT COLOUR FIX
       ====================================== */

    section[data-testid="stMain"] .stMarkdown p,
    section[data-testid="stMain"] .stMarkdown li,
    section[data-testid="stMain"] .stMarkdown,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #263238 !important;
    }

    /* Dark navy headings */

    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4 {
        color: #172554 !important;
    }

    /* Hero section keeps white text */

    .hero,
    .hero * {
        color: #FFFFFF !important;
    }

    /* Real news result keeps green text */

    .real-box,
    .real-box * {
        color: #14532D !important;
    }

    /* Fake news result keeps red text */

    .fake-box,
    .fake-box * {
        color: #7F1D1D !important;
    }

    /* Sidebar Background */

    section[data-testid="stSidebar"] {
        background-color: #E2E8F0;
    }

    /* Sidebar Text */

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #172554;
    }

    /* Buttons */

    .stButton > button {
        background-color: #2563EB;
        color: #FFFFFF;
        border: 1px solid #2563EB;
        border-radius: 9px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
        border-color: #1D4ED8;
    }

    /* Input Fields */

    .stTextInput input,
    .stTextArea textarea {
        background-color: #FFFFFF;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = os.path.join(
    "models",
    "fake_news_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True

except Exception as error:
    model = None
    model_loaded = False


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "headline" not in st.session_state:
    st.session_state.headline = ""

if "article" not in st.session_state:
    st.session_state.article = ""


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📰 Smart News AI")

    st.caption("Machine Learning Based News Classification")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 News Detection",
            "📊 Prediction History",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Dataset:** WELFake")
    st.write("**Language:** Python")

    st.divider()

    st.caption("Final Year Major Project")


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">SMART FAKE NEWS DETECTION SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">An NLP and Machine Learning Based News Classification Platform</div>',
    unsafe_allow_html=True
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h2>Welcome to Smart Fake News Detection</h2>
        <p>
        Analyze news content using Natural Language Processing
        and Machine Learning. Enter a headline and article to
        receive a model-based prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">94.80%</div>
            <div class="metric-label">Logistic Regression Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">15,000</div>
            <div class="metric-label">Maximum TF-IDF Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">2</div>
            <div class="metric-label">Models Evaluated</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### How the System Works")

    step1, step2, step3, step4 = st.columns(4)

    step1.info("**1. Input News**\n\nEnter headline and article.")
    step2.info("**2. Text Processing**\n\nTF-IDF converts text into features.")
    step3.info("**3. ML Prediction**\n\nLogistic Regression analyzes the text.")
    step4.info("**4. View Result**\n\nSee predicted class and confidence.")

    st.markdown("### Start Analyzing")

    if st.button("🔍 Go to News Detection", use_container_width=True):
        st.session_state.go_to_detection = True
        st.rerun()

    st.caption(
        "The displayed accuracy is from the held-out test evaluation "
        "during model development."
    )


# ==========================================
# NEWS DETECTION
# ==========================================

elif page == "🔍 News Detection":

    st.subheader("🔍 Analyze a News Article")

    st.write(
        "Enter the news headline and complete article below."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Load Sample News"):

            st.session_state.headline = (
                "Government Announces New Public Transportation Plan"
            )

            st.session_state.article = (
                "The government announced a new public transportation "
                "plan on Monday. Officials said the plan includes improving "
                "bus services, adding routes, and upgrading stations. "
                "The proposal will be reviewed by local authorities "
                "before implementation."
            )

            st.rerun()

    with col2:

        if st.button("Clear Inputs"):
            st.session_state.headline = ""
            st.session_state.article = ""
            st.rerun()

    headline = st.text_input(
        "News Headline",
        key="headline",
        placeholder="Enter the news headline..."
    )

    article = st.text_area(
        "News Article",
        key="article",
        height=220,
        placeholder="Paste the complete news article here..."
    )

    if st.button(
        "🚀 Analyze News",
        type="primary",
        use_container_width=True
    ):

        if not model_loaded:

            st.error(
                "Model could not be loaded. Check whether "
                "models/fake_news_model.joblib exists."
            )

        elif not headline.strip() or not article.strip():

            st.warning(
                "Please enter both the headline and article."
            )

        else:

            combined_text = headline.strip() + " " + article.strip()

            try:

                prediction = model.predict([combined_text])[0]

                probabilities = model.predict_proba(
                    [combined_text]
                )[0]

                classes = list(model.classes_)
                predicted_index = classes.index(prediction)

                confidence = probabilities[predicted_index] * 100

                # WELFake label mapping:
                # 0 = Fake
                # 1 = Real

                if int(prediction) == 1:
                    result = "LIKELY REAL NEWS"
                    result_class = "real-box"
                    icon = "✅"

                else:
                    result = "LIKELY FAKE NEWS"
                    result_class = "fake-box"
                    icon = "⚠️"

                st.markdown("---")

                st.subheader("Prediction Result")

                st.markdown(
                    f"""
                    <div class="result-box {result_class}">
                        <h1>{icon}</h1>
                        <h2>{result}</h2>
                        <h3>Model Confidence: {confidence:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(float(confidence / 100))

                st.markdown("### Prediction Details")

                detail1, detail2 = st.columns(2)

                with detail1:
                    st.metric(
                        "Predicted Class",
                        "Real" if int(prediction) == 1 else "Fake"
                    )

                with detail2:
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.2f}%"
                    )

                st.info(
                    "Confidence represents the model's estimated "
                    "probability for its predicted class. It does not "
                    "prove that the news is factually true or false."
                )

                st.session_state.history.append({
                    "Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                    "Headline": headline,
                    "Prediction": (
                        "Real" if int(prediction) == 1 else "Fake"
                    ),
                    "Confidence": round(confidence, 2)
                })

            except Exception as error:

                st.error(f"Prediction error: {error}")


# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "📊 Prediction History":

    st.subheader("📊 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.metric(
            "Total Articles Analyzed",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv,
            file_name="news_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.rerun()

    else:

        st.info(
            "No predictions yet. Visit News Detection to analyze an article."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About the Project")

    st.markdown("""
    ### Smart Fake News Detection System

    This project is developed using Natural Language Processing
    and Machine Learning techniques to classify news articles
    based on their textual content.

    ### Project Objectives

    - Analyze news headlines and article content.
    - Process textual information using TF-IDF.
    - Train and evaluate machine learning classifiers.
    - Predict whether an article is likely fake or real.
    - Present results through an interactive web application.

    ### Machine Learning Models

    1. Logistic Regression
    2. Multinomial Naive Bayes

    ### Selected Model

    Logistic Regression achieved 94.80% accuracy on the
    held-out test dataset and was selected for the application.

    ### Technology Stack

    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Streamlit
    - Joblib

    ### Dataset

    WELFake Dataset
    """)

    st.warning(
        "This application is an academic machine learning project. "
        "It cannot independently verify the factual accuracy of news. "
        "Always cross-check important information with reliable sources."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="footer">
    Smart Fake News Detection System |
    Final Year Major Project |
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)
