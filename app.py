import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("daybue_data_machinelearning.csv")  # Load your dataset
    return df

df = load_data()

# Extract dropdown options dynamically
mutation_categories = df["mutation_category"].dropna().unique().tolist()
variants = df["variant"].dropna().unique().tolist()
age_categories = df["age_category"].dropna().unique().tolist()
titrate_options = df["titrate"].dropna().unique().tolist()
epilepsy_options = df["epilepsy_history"].dropna().unique().tolist()

# Define features & target
features = ["mutation_category", "variant", "age_category", "titrate", "dosage", "rsbq_baseline", "cgis", "epilepsy_history"]
target = "total_improvement"

# Preprocess data
df = df.dropna(subset=[target])  # Remove rows with missing target values
for col in ["mutation_category", "variant", "age_category", "titrate", "epilepsy_history"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Daybue Response Predictor")
st.write("Enter patient details below to predict response to treatment.")

# User input fields with dropdowns dynamically populated from dataset
mutation_category = st.selectbox("Mutation Category", mutation_categories)
variant = st.selectbox("Variant", variants)
age_category = st.selectbox("Age Category", age_categories)
titrate = st.selectbox("Titration Method", titrate_options)
dosage = st.slider("Dosage (1-5)", min_value=1, max_value=5, value=3)
rsbq_baseline = st.number_input("RSBQ Baseline Score", min_value=0, max_value=100, value=50)
cgis = st.number_input("Clinical Global Impression of Severity (CGIS)", min_value=1, max_value=7, value=4)
epilepsy_history = st.selectbox("Epilepsy History", epilepsy_options)

# Convert inputs to DataFrame
input_data = pd.DataFrame([[mutation_category, variant, age_category, titrate, dosage, rsbq_baseline, cgis, epilepsy_history]],
                          columns=features)

# Encode categorical variables
for col in ["mutation_category", "variant", "age_category", "titrate", "epilepsy_history"]:
    input_data[col] = LabelEncoder().fit_transform(input_data[col].astype(str))

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Total Improvement: {round(prediction, 2)}")
