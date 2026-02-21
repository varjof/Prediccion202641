import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Pre-trained Objects ---
onehot_encoder = joblib.load('onehot_encoder.joblib')
minmax_scaler = joblib.load('minmax_scaler.joblib')
logistic_regression_model = joblib.load('logistic_regression_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# --- 2. Define Streamlit Layout and Inputs ---
st.title('Predicción de Aprobación de Curso')

# Input for 'Felder'
felder_options = onehot_encoder.categories_[0].tolist()
selected_felder = st.selectbox('Seleccione su estilo de aprendizaje (Felder):', felder_options)

# Input for 'Examen_admisión'
examen_admision_value = st.number_input(
    'Ingrese su puntaje en el examen de admisión:',
    min_value=3.00,
    max_value=5.83,
    value=4.00, # Default value within the range
    step=0.01
)

predict_button = st.button('Predecir')

# --- 3. Prediction Logic (on button click) ---
if predict_button:
    # Preprocess 'Felder' input
    input_felder = pd.DataFrame([[selected_felder]], columns=['Felder'])
    felder_encoded_input = onehot_encoder.transform(input_felder)
    encoded_feature_names = onehot_encoder.get_feature_names_out(['Felder'])
    felder_df_input = pd.DataFrame(felder_encoded_input, columns=encoded_feature_names)

    # Preprocess 'Examen_admisión' input
    input_examen_admision = pd.DataFrame([[examen_admision_value]], columns=['Examen_admisión'])
    examen_admision_scaled_input = minmax_scaler.transform(input_examen_admision)
    examen_df_input = pd.DataFrame(examen_admision_scaled_input, columns=['Examen_admisión_scaled'])

    # Concatenate all preprocessed features
    processed_input_df = pd.concat([felder_df_input, examen_df_input], axis=1)
    
    # Reorder columns to match the model's expected feature order
    processed_input_df = processed_input_df[logistic_regression_model.feature_names_in_]

    # Make prediction
    prediction = logistic_regression_model.predict(processed_input_df)

    # Decode the prediction
    decoded_prediction = label_encoder.inverse_transform(prediction)

    # --- 4. Display Prediction ---
    st.success(f'La predicción de aprobación del curso es: {decoded_prediction[0]}')
