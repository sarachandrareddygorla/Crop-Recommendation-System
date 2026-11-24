
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "crop_recommendation_model.joblib")
DATA_FILE = os.path.join(BASE_DIR, "Crop_recommendation.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "model_results.csv")

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(180deg, #f0f8f0 0%, #ffffff 65%);
    color: #1f2937;
}

/* Main text */
[data-testid="stAppViewContainer"] {
    color: #1f2937;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stWidgetLabel"] {
    color: #1f2937 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e4f2e4;
}

[data-testid="stSidebar"] * {
    color: #173b25 !important;
}

/* Hero banner */
.hero {
    padding: 1.8rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #14532d, #23834a);
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
}

.hero h1 {
    color: white !important;
    margin-bottom: 0.4rem;
}

.hero p {
    color: #e8f5e9 !important;
    font-size: 1.05rem;
}

/* Prediction result */
.result {
    padding: 1.6rem;
    border-radius: 16px;
    text-align: center;
    background: #e8f5e9;
    border: 1px solid #81c784;
    margin-top: 1rem;
}

.result p {
    color: #245332 !important;
}

.result h2 {
    color: #14532d !important;
    font-size: 2rem;
}

/* Buttons */
div.stButton > button[kind="primary"] {
    background-color: #19733a;
    border-color: #19733a;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #14532d;
    border-color: #14532d;
    color: white !important;
}

/* Input fields */
.stNumberInput input {
    color: #1f2937 !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #d1e5d1;
    border-radius: 10px;
}

/* Section headings */
h1, h2, h3, h4 {
    color: #14532d;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #c8dfc8;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def get_data():
    return pd.read_csv(DATA_FILE)


def show_setup_error():

    missing = [
        f for f in [MODEL_FILE, DATA_FILE]
        if not os.path.exists(f)
    ]

    if missing:
        st.error(
            "Required project file(s) not found: "
            + ", ".join(os.path.basename(f) for f in missing)
        )

        st.info(
            "Run the complete notebook, including its final export cell, "
            "and launch Streamlit from the project folder."
        )

        st.stop()


show_setup_error()

model = get_model()
data = get_data()

if not set(FEATURES + ["label"]).issubset(data.columns):
    st.error(
        "The dataset does not contain the expected feature columns and label."
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🌱 CropWise")

st.sidebar.caption(
    "Machine Learning Based Crop Recommendation"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Crop Prediction",
        "Dataset Explorer",
        "Model Performance",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Academic project prototype")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🌱 Crop Recommendation System</h1>
        <p>
            Make an informed crop choice using soil nutrients
            and weather conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "This application uses a supervised machine-learning model "
        "trained on labelled crop data. Enter soil and environmental "
        "measurements to obtain a predicted crop category."
    )

    a, b, c = st.columns(3)

    a.metric("Dataset Records", f"{len(data):,}")
    b.metric("Crop Categories", data["label"].nunique())
    c.metric("Model Inputs", len(FEATURES))

    st.subheader("How to use the application")

    st.markdown("""
    1. Open **Crop Prediction** from the sidebar.
    2. Enter nitrogen, phosphorus and potassium values.
    3. Enter temperature, humidity, soil pH and rainfall.
    4. Select **Recommend Crop** to view the prediction.
    """)

    st.info(
        "This is an educational decision-support prototype, "
        "not a substitute for local soil testing or advice "
        "from an agricultural expert."
    )


# --------------------------------------------------
# CROP PREDICTION PAGE
# --------------------------------------------------

elif page == "Crop Prediction":

    st.markdown("""
    <div class="hero">
        <h1>🌾 Crop Prediction</h1>
        <p>
            Enter the measured soil and weather conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Soil Nutrient Values")

    col1, col2, col3 = st.columns(3)

    values = {}

    for col, feature, label in [
        (col1, "N", "Nitrogen (N)"),
        (col2, "P", "Phosphorus (P)"),
        (col3, "K", "Potassium (K)")
    ]:

        lo = float(data[feature].min())
        hi = float(data[feature].max())
        default = float(data[feature].median())

        values[feature] = col.number_input(
            label,
            min_value=lo,
            max_value=hi,
            value=default,
            key=feature
        )

    st.subheader("Weather and Soil Conditions")

    col4, col5 = st.columns(2)

    for col, feature, label in [
        (col4, "temperature", "Temperature (°C)"),
        (col5, "humidity", "Humidity (%)"),
        (col4, "ph", "Soil pH"),
        (col5, "rainfall", "Rainfall (mm)")
    ]:

        lo = float(data[feature].min())
        hi = float(data[feature].max())
        default = float(data[feature].median())

        values[feature] = col.number_input(
            label,
            min_value=lo,
            max_value=hi,
            value=default,
            key=feature
        )

    st.caption(
        "Input limits are based on the minimum and maximum "
        "values in the supplied training dataset."
    )

    if st.button(
        "🌱 Recommend Crop",
        type="primary",
        use_container_width=True
    ):

        row = pd.DataFrame(
            [[values[f] for f in FEATURES]],
            columns=FEATURES
        )

        prediction = model.predict(row)[0]

        st.markdown(
            f"""
            <div class="result">
                <p>Recommended crop predicted by the model</p>
                <h2>🌿 {str(prediction).title()}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("View Entered Values"):
            st.dataframe(row, use_container_width=True)


# --------------------------------------------------
# DATASET EXPLORER PAGE
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.markdown("""
    <div class="hero">
        <h1>📊 Dataset Explorer</h1>
        <p>
            Explore the dataset used to train the crop classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    m1.metric("Records", f"{len(data):,}")
    m2.metric("Features", len(FEATURES))
    m3.metric("Crop Classes", data["label"].nunique())

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(15),
        use_container_width=True
    )

    st.subheader("Crop Class Distribution")

    counts = (
        data["label"]
        .value_counts()
        .rename_axis("Crop")
        .reset_index(name="Records")
    )

    fig = px.bar(
        counts,
        x="Crop",
        y="Records",
        color="Records",
        color_continuous_scale="Greens"
    )

    fig.update_layout(
        xaxis_title="Crop",
        yaxis_title="Number of Records",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numerical Feature Summary")

    st.dataframe(
        data[FEATURES].describe().T,
        use_container_width=True
    )

    st.subheader("Feature Correlation")

    corr = data[FEATURES].corr(numeric_only=True)

    fig2 = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1
    )

    fig2.update_layout(template="plotly_white")

    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>📈 Model Performance</h1>
        <p>
            Evaluation metrics generated when the notebook is run.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(RESULTS_FILE):

        results = pd.read_csv(RESULTS_FILE)

        st.subheader("Model Evaluation Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        metric_options = [
            c for c in [
                "Accuracy",
                "Precision (macro)",
                "Recall (macro)",
                "F1-score (macro)"
            ]
            if c in results.columns
        ]

        if (
            metric_options
            and "Model" in results.columns
            and not results.empty
        ):

            metric = st.selectbox(
                "Choose Evaluation Metric",
                metric_options
            )

            fig = px.bar(
                results,
                x="Model",
                y=metric,
                color="Model",
                range_y=[0, 1],
                title=f"{metric} by Model",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if (
            "F1-score (macro)" in results.columns
            and "Model" in results.columns
            and not results.empty
        ):

            best = results.sort_values(
                "F1-score (macro)",
                ascending=False
            ).iloc[0]

            st.success(
                "Best model by macro F1-score in the latest notebook run: "
                + str(best["Model"])
            )

    else:

        st.warning(
            "Model metrics file is not available yet. "
            "Run the full notebook and its export cell."
        )

    st.subheader("Algorithms Included")

    st.markdown("""
    - Decision Tree Classifier
    - Random Forest Classifier
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine classification using SVC
    """)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About the Project</h1>
        <p>
            Crop Recommendation System using supervised machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Objective")

    st.write(
        "To predict a suitable crop category from seven soil and "
        "weather inputs using a model trained on labelled crop records."
    )

    st.subheader("Input Parameters")

    st.markdown("""
    - **N, P, K:** Soil nitrogen, phosphorus and potassium values
    - **Temperature:** Ambient temperature
    - **Humidity:** Relative humidity
    - **pH:** Soil acidity/alkalinity measure
    - **Rainfall:** Rainfall measurement
    """)

    st.subheader("Technology Stack")

    st.markdown(
        "Python · Pandas · Scikit-learn · Joblib · Streamlit · Plotly"
    )

    st.warning(
        "A model prediction is not a guarantee of yield or profitability. "
        "Real planting decisions should consider local conditions, soil tests, "
        "water availability, season and professional agricultural guidance."
    )
