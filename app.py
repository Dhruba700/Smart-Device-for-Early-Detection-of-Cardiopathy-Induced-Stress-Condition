import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import streamlit as st

# Set page configuration FIRST
st.set_page_config(
    page_title="Cardiopathy Prediction App",
    page_icon="🫀",
    layout="wide"
)

# Check for PyPDF2
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PyPDF2 module not found. PDF support will be disabled. Please install PyPDF2 using 'pip install PyPDF2'.")

# Load or train the model with evaluation
@st.cache_resource
def load_or_train_model(df):
    try:
        model = joblib.load('cardiopathy_model.pkl')
        scaler = joblib.load('cardiopathy_scaler.pkl')
    except:
        X = df.drop('Cardiopathy', axis=1)
        y = df['Cardiopathy']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        with st.expander("Model Evaluation Metrics"):
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Classification Report:")
            st.text(report)
        
        joblib.dump(model, 'cardiopathy_model.pkl')
        joblib.dump(scaler, 'cardiopathy_scaler.pkl')
    
    return model, scaler

# Prediction function
def predict_cardiopathy(model, scaler, input_data):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    feature_importance = dict(zip(['Temp', 'ECG', 'Stress', 'SpO2', 'BPM'], model.feature_importances_))
    return prediction, probability, feature_importance

# Improved mock data extraction with Health Ninja context
def extract_data_from_image(image):
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    return {
        'Temp': 36.5 + (height % 10) * 0.1,  # Body temperature
        'ECG': 400 + (width % 200),          # ECG signal
        'Stress': 30 + (np.mean(img_array) % 50),  # Stress level
        'SpO2': 92 + (height % 8),           # Oxygen saturation
        'BPM': 60 + (width % 60)             # Heart rate
    }

def extract_data_from_video(video_bytes):
    return {
        'Temp': np.random.uniform(36.0, 38.0),
        'ECG': np.random.uniform(300, 600),
        'Stress': np.random.randint(20, 80),
        'SpO2': np.random.uniform(90, 100),
        'BPM': np.random.randint(60, 120)
    }

def extract_data_from_pdf(pdf_file):
    if not PDF_AVAILABLE:
        return {
            'Temp': np.random.uniform(36.0, 38.0),
            'ECG': np.random.uniform(300, 600),
            'Stress': np.random.randint(20, 80),
            'SpO2': np.random.uniform(90, 100),
            'BPM': np.random.randint(60, 120)
        }
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    # Attempt to find health-related terms (placeholder logic)
    data = {
        'Temp': 36.5,
        'ECG': 400,
        'Stress': 30,
        'SpO2': 92,
        'BPM': 60
    }
    if "heart rate" in text.lower():
        data['BPM'] = 60 + (len(text) % 60)
    if "stress" in text.lower():
        data['Stress'] = 30 + (len(text) % 50)
    return data

# Display results with manual override
def display_results(prediction, probability, feature_importance, input_data):
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"Cardiopathy Detected! Probability: {probability[1]*100:.2f}%")
        else:
            st.success(f"No Cardiopathy Detected. Probability: {probability[0]*100:.2f}%")
    
    with col2:
        st.write("Extracted Values (Editable):")
        edited_data = {}
        for key, value in input_data.items():
            edited_data[key] = st.number_input(f"{key}", value=float(value), key=f"edit_{key}")
    
    st.warning("Note: Values are currently mock data based on 'Health Ninja' context. Implement actual sensor data extraction for accurate results.")
    
    if edited_data != input_data:
        prediction, probability, feature_importance = predict_cardiopathy(model, scaler, edited_data)
        st.write("Updated Prediction with Edited Values:")
        if prediction == 1:
            st.error(f"Cardiopathy Detected! Probability: {probability[1]*100:.2f}%")
        else:
            st.success(f"No Cardiopathy Detected. Probability: {probability[0]*100:.2f}%")
    
    st.subheader("Detailed Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(['No Cardiopathy', 'Cardiopathy'], probability, color=['green', 'red'])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Probability')
        ax1.set_title('Prediction Probability')
        st.pyplot(fig1)
    
    with col4:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        features, importance = zip(*feature_importance.items())
        ax2.bar(features, importance, color='blue')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Feature Importance')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# Main app
def main():
    global model, scaler
    
    st.title("🫀 Cardiopathy Prediction App - The Health Ninja")
    st.markdown("""
    Upload your medical reports (CSV, Photo, Video, or PDF) to predict cardiopathy.
    First, upload a training CSV file.
    """)
    
    tab_names = ["Prediction", "About"] if PDF_AVAILABLE else ["Prediction", "About"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Prediction Tab
        st.subheader("Upload Training Data")
        training_file = st.file_uploader(
            "Upload a CSV file to train the model (must include: Temp, ECG, Stress, SpO2, BPM, Cardiopathy)", 
            type=['csv'],
            key="training_data_uploader"
        )
        
        if training_file is None:
            st.warning("Please upload a training dataset to proceed.")
            return
        
        df = pd.read_csv(training_file)
        required_cols = ['Temp', 'ECG', 'Stress', 'SpO2', 'BPM', 'Cardiopathy']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Training CSV must contain columns: {', '.join(required_cols)}")
            return
        
        model, scaler = load_or_train_model(df)
        st.success("Model trained successfully!")
        
        st.subheader("Upload Medical Report")
        upload_tabs = st.tabs(["CSV", "Photo", "Video", "PDF"] if PDF_AVAILABLE else ["CSV", "Photo", "Video"])
        
        with upload_tabs[0]:  # Upload CSV
            uploaded_csv = st.file_uploader(
                "Upload a CSV medical report (Temp, ECG, Stress, SpO2, BPM)",
                type=['csv'],
                key="csv_uploader"
            )
            if uploaded_csv is not None:
                pred_df = pd.read_csv(uploaded_csv)
                st.write("Uploaded Data Preview:", pred_df)
                if all(col in pred_df.columns for col in required_cols[:-1]):
                    input_data = pred_df.iloc[0][required_cols[:-1]].to_dict()
                    if st.button("Predict from CSV", key="csv_predict"):
                        prediction, probability, feature_importance = predict_cardiopathy(model, scaler, input_data)
                        display_results(prediction, probability, feature_importance, input_data)
                else:
                    st.error(f"CSV must contain columns: {', '.join(required_cols[:-1])}")

        with upload_tabs[1]:  # Upload Photo
            uploaded_photo = st.file_uploader(
                "Upload a photo of your medical report",
                type=['jpg', 'png', 'jpeg'],
                key="photo_uploader"
            )
            if uploaded_photo is not None:
                image = Image.open(uploaded_photo)
                st.image(image, caption="Uploaded Medical Report", use_column_width=True)
                input_data = extract_data_from_image(image)
                if st.button("Predict from Photo", key="photo_predict"):
                    prediction, probability, feature_importance = predict_cardiopathy(model, scaler, input_data)
                    display_results(prediction, probability, feature_importance, input_data)

        with upload_tabs[2]:  # Upload Video
            uploaded_video = st.file_uploader(
                "Upload a video containing medical data",
                type=['mp4', 'avi'],
                key="video_uploader"
            )
            if uploaded_video is not None:
                st.video(uploaded_video)
                input_data = extract_data_from_video(uploaded_video)
                if st.button("Predict from Video", key="video_predict"):
                    prediction, probability, feature_importance = predict_cardiopathy(model, scaler, input_data)
                    display_results(prediction, probability, feature_importance, input_data)

        if PDF_AVAILABLE:
            with upload_tabs[3]:  # Upload PDF
                uploaded_pdf = st.file_uploader(
                    "Upload a PDF medical report",
                    type=['pdf'],
                    key="pdf_uploader"
                )
                if uploaded_pdf is not None:
                    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
                    first_page_text = pdf_reader.pages[0].extract_text()
                    st.text_area("PDF Preview (First Page)", first_page_text, height=200)
                    input_data = extract_data_from_pdf(uploaded_pdf)
                    if st.button("Predict from PDF", key="pdf_predict"):
                        prediction, probability, feature_importance = predict_cardiopathy(model, scaler, input_data)
                        display_results(prediction, probability, feature_importance, input_data)

    with tabs[1]:  # About Tab
        st.subheader("About The Health Ninja")
        st.markdown("""
        ### Project Details
        **Title:** "THE HEALTH NINJA" - Smart device for early detection of cardiopathy and induced stress condition  
        **Team Name:** MEDVENGERS  
        **Institution:** Department of Biomedical Engineering, JIS Group Educational Initiatives, Institution's Innovation Council  
        **Team Members:** Anwita Ghosh, Disha bhandari, Dhrubajoti Adhikari, Atmika Paul, Ranajit Ghosh, Bidipta Chakrabarti 

        ### Introduction
        Health Ninja is a personalized health monitoring system that tracks vital signs and detects stress in real-time. Using advanced biosensors and smart analytics, it provides proactive insights to prevent cardiopathy. Designed for effortless integration, it redefines personal healthcare with precision and ease.

        ### Novelty
        - Real-time detection of stress levels and cardiovascular abnormalities
        - Combines ECG, biofeedback skin sensors, and optical sensors for comprehensive analysis

        ### Results
        - Successful real-time health tracking
        - Accurate detection of stress & heart rate variations
        - User-friendly interface & seamless operation

        ### Future Scope
        - Integration with smartwatch
        - AI-driven predictive analytics
        - Enhanced cloud-based remote monitoring

        ### Acknowledgement
        We extend our heartfelt gratitude to our mentors, Dr. Karabi Ganguly and Dr. Dibyendu Mondal, for their invaluable guidance and encouragement throughout this project. We also thank our faculty, project collaborators, and the organizers of JISTECH for providing us with the platform to showcase our innovation.
        """)

if __name__ == "__main__":
    main()