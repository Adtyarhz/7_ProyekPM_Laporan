import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# --- Load data & model artefacts ---
@st.cache_data
def load_data():
    return pd.read_csv('car_price.csv')

@st.cache_resource
def load_model(model_file='models.pkl'):
    return joblib.load(model_file)

# --- Load model and data ---
df = load_data()
art = load_model()

reg        = art['reg']
clf        = art['clf']
scaler     = art['scaler']
encoders   = art['encoders']
kmeans     = art['kmeans']
median_val = art['median_val']
target_col = art['target_col']

name_options = sorted(df['name'].unique().tolist())

# --- Sidebar Menu ---
st.sidebar.markdown(
    """
    <style>
    .sidebar .st-emotion-cache-16txtl3 {
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("https://img.icons8.com/clouds/100/car.png", use_container_width=True)
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Predict Price"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by Group 7")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Style for selectbox background */
    .stSelectbox>div>div>div {
        background-color: #e3f2fd !important; /* Light Blue */
        color: #262730; /* Dark Text */
    }
    .stSelectbox>div>div>div:focus {
        border-color: #1a73e8 !important; /* Focus border */
    }

    /* Style for number input background */
    .stNumberInput>div>div>div>input {
        background-color: #e3f2fd !important; /* Light Blue */
        color: #262730; /* Dark Text */
    }
    .stNumberInput>div>div>div>input:focus {
        border-color: #1a73e8 !important; /* Focus border */
    }
    
    /* Style for the predict button background */
    .stButton>button {
        background-color: #1a73e8 !important; /* Google Blue */
        color: white !important;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0d47a1 !important; /* Darker Blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- Home Page ---
if menu == "🏠 Home":
    st.title("🔮 Welcome to Car Price Predictor")
    st.markdown("""
        Predict the estimated **selling price** of a used car based on its specifications.

        This app uses a **Machine Learning model** trained on historical car data to make predictions based on:
        - Car brand and model
        - Year of manufacture
        - Fuel type and transmission
        - Ownership and mileage

        ---
        """)
    st.image("https://img.freepik.com/free-vector/car-race-illustration_1284-24744.jpg", use_container_width=True)
    st.info("Go to the '📊 Predict Price' section from the sidebar to try it out!")

# --- Prediction Page ---
elif menu == "📊 Predict Price":
    st.title("📊 Car Price Prediction Form")
    st.markdown("Please fill in your car details:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            name         = st.selectbox("Car Name", name_options)
            year         = st.number_input("Year", min_value=int(df['year'].min()),
                                           max_value=int(df['year'].max()),
                                           value=int(df['year'].median()))
            fuel         = st.selectbox("Fuel Type", sorted(df['fuel'].unique().tolist()))
            transmission = st.selectbox("Transmission", sorted(df['transmission'].unique().tolist()))
        with col2:
            km_driven    = st.number_input("Kilometers Driven", min_value=0,
                                           value=int(df['km_driven'].median()))
            seller_type  = st.selectbox("Seller Type", sorted(df['seller_type'].unique().tolist()))
            owner        = st.selectbox("Owner", sorted(df['owner'].unique().tolist()))

        submitted = st.form_submit_button("🚗 Predict Now")

    if submitted:
        with st.spinner("Making prediction..."):
            df_new = pd.DataFrame([{
                'name': name,
                'year': year,
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner
            }])

            # Encode & scale
            for col, le in encoders.items():
                df_new[col] = le.transform(df_new[col].astype(str))
            Xs = scaler.transform(df_new)

            # Predict
            price_pred   = reg.predict(Xs)[0]
            label_pred   = clf.predict(Xs)[0]
            cluster_pred = kmeans.predict(Xs)[0]

            # Results
            st.success("✅ Prediction Successful!")
            st.subheader("📊 Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Estimated Price", f"₹{price_pred:,.0f}")
            col2.metric("🏷️ Price Label", "High" if label_pred == 1 else "Low")
            col3.metric("🔢 Cluster Group", str(cluster_pred))

            with st.expander("Model Details"):
                st.write(f"Regression Model: {reg.__class__.__name__}")
                st.write(f"Classification Model: {clf.__class__.__name__}")
                st.write(f"Cluster Model: {clf.__class__.__name__}")