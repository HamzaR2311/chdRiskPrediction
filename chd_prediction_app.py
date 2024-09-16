import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os


# Set page configuration
st.set_page_config(page_title="CHD Risk Prediction Tool", layout="wide")

# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("framingham.csv")
        df = df.drop(['education', 'currentSmoker'], axis=1)
        df.dropna(axis=0, inplace=True)
        df.rename(columns={"male": "sex"}, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'framingham.csv' is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        return None

# Preprocess the data
def preprocess_data(df):
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    
    # Balance the dataset using SMOTE and RandomUnderSampler
    before = dict(Counter(y))
    
    over_sampling = SMOTE(sampling_strategy=0.7, random_state=42)
    under_sampling = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
    steps = [("o", over_sampling), ("u", under_sampling)]
    pipeline = Pipeline(steps=steps)
    
    X_balanced, y_balanced = pipeline.fit_resample(X, y)
    
    after = dict(Counter(y_balanced))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    return X_scaled, y_balanced, scaler, X.columns, before, after

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    
    return rf_model, accuracy, precision, recall, f1, auc, cv_scores

def create_risk_visualization(prediction):
    risk_level = "Low" if prediction < 0.10 else "Moderate" if prediction < 0.20 else "High"
    color = "green" if risk_level == "Low" else "orange" if risk_level == "Moderate" else "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "CHD Risk", 'font': {'size': 24, 'color': color}},
        delta = {'reference': 10, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': 'rgba(0, 250, 0, 0.1)'},
                {'range': [10, 20], 'color': 'rgba(250, 250, 0, 0.1)'},
                {'range': [20, 100], 'color': 'rgba(250, 0, 0, 0.1)'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20}}))
    
    fig.update_layout(
        height=300,
        font={'color': color, 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig, risk_level, color

# Create feature importance plot
def create_feature_importance_plot(model, feature_names):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                 title='Top 10 Feature Importances')
    fig.update_layout(height=400)
    return fig

# Main function
def main():
    
    st.title("10-Year CHD Risk Prediction Tool")
    st.write("This tool predicts the 10-year risk of coronary heart disease (CHD) based on patient information.")
    
    # Load and preprocess data
    df = load_data()
    if df is None:
        return
    
    X_scaled, y_balanced, scaler, feature_names, _, _ = preprocess_data(df)
    
    # Train model or load pre-trained model
    model_file = 'chd_model.joblib'
    metrics_file = 'model_metrics.joblib'
    
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        st.sidebar.success("Loaded pre-trained model")
        
        if os.path.exists(metrics_file):
            metrics = joblib.load(metrics_file)
        else:
            st.sidebar.warning("Metrics file not found. Recalculating metrics...")
            metrics = calculate_metrics(model, X_scaled, y_balanced)
            joblib.dump(metrics, metrics_file)
    else:
        model, metrics = train_model(X_scaled, y_balanced)
        joblib.dump(model, model_file)
        joblib.dump(metrics, metrics_file)
        st.sidebar.success("Trained and saved new model")
    
    # User input form
    st.header("Patient Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sex = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=70, value=0)
        BPMeds = 1 if st.selectbox("BP Medication", ["No", "Yes"]) == "Yes" else 0
        prevalentStroke = 1 if st.selectbox("Prevalent Stroke", ["No", "Yes"]) == "Yes" else 0
    
    with col2:
        prevalentHyp = 1 if st.selectbox("Prevalent Hypertension", ["No", "Yes"]) == "Yes" else 0
        diabetes = 1 if st.selectbox("Diabetes", ["No", "Yes"]) == "Yes" else 0
        totChol = st.number_input("Total Cholesterol", min_value=100, max_value=600, value=200)
        sysBP = st.number_input("Systolic BP", min_value=80, max_value=300, value=120)
        diaBP = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
    
    with col3:
        BMI = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        heartRate = st.number_input("Heart Rate", min_value=40, max_value=150, value=75)
        glucose = st.number_input("Glucose", min_value=40, max_value=400, value=80)
    
    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'sex': [sex], 'age': [age], 'cigsPerDay': [cigsPerDay], 'BPMeds': [BPMeds],
        'prevalentStroke': [prevalentStroke], 'prevalentHyp': [prevalentHyp], 'diabetes': [diabetes],
        'totChol': [totChol], 'sysBP': [sysBP], 'diaBP': [diaBP], 'BMI': [BMI],
        'heartRate': [heartRate], 'glucose': [glucose]
    })
    
    # Ensure the order of columns matches the training data
    user_data = user_data[feature_names]
    
    # Scale user input
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    if st.button("Predict CHD Risk"):
        try:
            prediction = model.predict_proba(user_data_scaled)[0][1]
            
            # Display results
            st.header("CHD Risk Prediction Results")
            
            # Improved risk visualization
            fig, risk_level, color = create_risk_visualization(prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{risk_level} Risk</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{prediction:.2%} chance of developing CHD in the next 10 years.</h2>", unsafe_allow_html=True)
            
            st.write("Please note that this is a prediction based on the model and should not be considered as definitive medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.")
            
            # Recommendations based on risk factors
            st.subheader("Recommendations")
            if cigsPerDay > 0:
                st.write("- Consider a smoking cessation program to reduce CHD risk.")
            if BMI > 25:
                st.write("- A weight management program may be beneficial to reduce CHD risk.")
            if sysBP > 120 or diaBP > 80:
                st.write("- Monitor blood pressure regularly and consider lifestyle changes or medication as recommended by your doctor.")
            if totChol > 200:
                st.write("- Consider dietary changes and possibly medication to manage cholesterol levels.")
            if glucose > 100:
                st.write("- Monitor blood glucose levels and consult with a healthcare provider about diabetes management.")
            
        except Exception as e:
            st.error(f"An error occurred while making the prediction: {str(e)}")

if __name__ == "__main__":
    main()