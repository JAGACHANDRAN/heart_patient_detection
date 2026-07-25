import streamlit as st
import pandas as pd
import joblib

# 1. Load your saved heart disease model brain
# (Ensure 'heart_disease_model.pkl' and 'heart.csv' are in the same directory)
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

# 2. Design the Web Page Header
st.title("❤️ Heart Disease Clinical Prediction App")
st.write("Input patient clinical metrics below to calculate immediate heart disease risk indicators.")

st.markdown("---")

# 3. Create two visual columns for patient data entry
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Profile")
    age = st.slider("Patient Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Patient Sex", options=["Female", "Male"])
    
    # Convert Sex to 1 (Male) and 0 (Female) matching the dataset rules
    sex_encoded = 1 if sex == "Male" else 0
    
    chest_pain = st.selectbox("Chest Pain Type (cp)", options=[
        "0: Typical Angina", 
        "1: Atypical Angina", 
        "2: Non-anginal Pain", 
        "3: Asymptomatic"
    ])
    cp_encoded = int(chest_pain.split(":")[0])

with col2:
    st.subheader("Clinical Vital Signs")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
    
    fbs_encoded = 1 if fbs == "Yes" else 0

st.markdown("---")

# 4. Predict Button Logic
if st.button("Analyze Patient Risk Metrics", type="primary"):
    try:
        # Load one row from your original data to copy the remaining default columns 
        raw_data = pd.read_csv('heart.csv')
        X_template = raw_data.drop(columns=['target'])
        
        # Isolate a single baseline template row
        sample_row = X_template.iloc[[0]].copy()
        
        # Overwrite the template features with YOUR live website user inputs
        sample_row['age'] = age
        sample_row['sex'] = sex_encoded
        sample_row['cp'] = cp_encoded
        sample_row['trestbps'] = trestbps
        sample_row['chol'] = chol
        sample_row['fbs'] = fbs_encoded
        
        # Run inference through your saved classifier
        prediction = model.predict(sample_row)
        
        # 5. Display the output result with Feature Reasoning
        st.subheader("🎯 Diagnostic Output Result:")
        
        if prediction[0] == 1:
            st.error("🚨 **Warning: Indicators point to a HIGH RISK of Heart Disease.**")
            st.markdown("### 🔍 Contributing Risk Factors:")
            st.write("Based on the data provided, the following features are outside of optimal medical ranges and contributed to this risk assessment:")
            
            # Clinical heuristic checks for "Not Safe" reasoning
            if trestbps > 120:
                st.write(f"- **Blood Pressure:** {trestbps} mm Hg (Elevated; Normal is ≤ 120)")
            if chol > 200:
                st.write(f"- **Cholesterol:** {chol} mg/dl (High; Desirable is < 200)")
            if fbs_encoded == 1:
                st.write("- **Fasting Blood Sugar:** Elevated (> 120 mg/dl)")
            if cp_encoded > 0:
                st.write(f"- **Chest Pain:** Patient is experiencing {chest_pain.split(':')[1].strip()}")
            if age > 55:
                st.write(f"- **Age:** {age} (Risk naturally increases for patients over 55)")
                
        else:
            st.success("🎉 **Indicators point to a NORMAL evaluation. Low risk detected.**")
            st.markdown("### ✅ Healthy Indicators:")
            st.write("The model predicts safety largely because the following metrics are within healthy, optimal ranges:")
            
            # Clinical heuristic checks for "Safe" reasoning
            if trestbps <= 120:
                st.write(f"- **Blood Pressure:** {trestbps} mm Hg (Excellent, within normal limits)")
            if chol <= 200:
                st.write(f"- **Cholesterol:** {chol} mg/dl (Healthy range)")
            if fbs_encoded == 0:
                st.write("- **Fasting Blood Sugar:** Normal (< 120 mg/dl)")
            if cp_encoded == 0 or cp_encoded == 3:
                st.write("- **Chest Pain:** No concerning angina reported.")
            
    except FileNotFoundError:
        st.error("Error: Could not find 'heart.csv' or 'heart_disease_model.pkl'. Please ensure they are in the same folder as this script.")
    except Exception as e:
        st.error(f"Data alignment issue: {e}")