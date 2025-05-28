import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load('final_model.pkl')
deployment_features = joblib.load('final_features.pkl')

# Set page config for better layout
st.set_page_config(page_title="Churn Prediction App", page_icon="🔮", layout="wide")

# Apply custom CSS for background color and theme
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;  /* Light pastel blue */
        }
        .main-title {
            text-align: center;
            color: #0D47A1;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #333333;
            font-size: 18px;
            margin-top: 0;
        }
        .footer {
            text-align: center;
            color: #A9A9A9;
            font-size: 14px;
            margin-top: 50px;
        }
        .stButton>button {
            color: white;
            background-color: #0D47A1;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='main-title'> Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Fill out the details below to predict customer churn probability</p>", unsafe_allow_html=True)
st.write("---")

# Create two equal columns
col1, col2 = st.columns(2)

user_input = {}

# Balanced split of features into two columns
half = len(deployment_features) // 2
features_col1 = deployment_features[:half]
features_col2 = deployment_features[half:]

# First column inputs
with col1:
    for feature in features_col1:
        if feature in ['voice mail plan', 'international plan']:
            options = ['No', 'Yes']
            user_choice = st.selectbox(f"{feature.replace('_', ' ').title()}", options)
            user_input[feature] = 1 if user_choice == 'Yes' else 0
        else:
            user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0)

# Second column inputs
with col2:
    for feature in features_col2:
        if feature in ['voice mail plan', 'international plan']:
            options = ['No', 'Yes']
            user_choice = st.selectbox(f"{feature.replace('_', ' ').title()}", options)
            user_input[feature] = 1 if user_choice == 'Yes' else 0
        else:
            user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0)

st.write("---")

# Prediction button with new color theme
if st.button("Predict Churn"):
    input_df = pd.DataFrame([user_input])
    churn_prob = model.predict_proba(input_df)[:, 1][0]
    churn_pred = model.predict(input_df)[0]

    if churn_pred == 1:
        st.markdown("<h3 style='color: #FF5722;'>⚠️ Predicted Churn: Yes</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #2E7D32;'>✅ Predicted Churn: No</h3>", unsafe_allow_html=True)

    st.markdown(f"<h3 style='color: #0D47A1;'>Churn Probability: <strong>{churn_prob:.2f}</strong></h3>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 Mary's Project - Powered by Streamlit</div>", unsafe_allow_html=True)




