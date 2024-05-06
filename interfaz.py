import streamlit as st
import requests
import json
from pydantic import BaseModel

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
class DatosEntrada(BaseModel):
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int

# Título de la interfaz
st.title('Interfaz para predecir admisión')

# Formulario para ingresar los datos de entrada
st.write('Ingrese los datos de entrada:')
admission_type_id = st.number_input('Tipo de admisión')
discharge_disposition_id = st.number_input('Tipo de salida')
admission_source_id = st.number_input('Fuente de admisión')
time_in_hospital = st.number_input('Tiempo en el hospital')
num_lab_procedures = st.number_input('Número de procedimientos de laboratorio')
num_procedures = st.number_input('Número de procedimientos')
num_medications = st.number_input('Número de medicamentos')
number_outpatient = st.number_input('Número de consultas ambulatorias')
number_emergency = st.number_input('Número de emergencias')
number_inpatient = st.number_input('Número de ingresos hospitalarios')
number_diagnoses = st.number_input('Número de diagnósticos')

data = {
    "admission_type_id": admission_type_id,
    "discharge_disposition_id": discharge_disposition_id,
    "admission_source_id": admission_source_id,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses
}

# Botón para enviar los datos de entrada y obtener la predicción
if st.button('Predecir'):
    # Realizar una solicitud POST a la API
    response = requests.post('http://fastapi:8000/predecir/', headers=headers, data=json.dumps(data))
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        print(response.json())
        # Obtener la predicción del primer elemento de la lista "detail"
        prediccion = response.json()
        # Mostrar la predicción
        st.write('La predicción es:', prediccion)
    else:
        st.write('Error al realizar la predicción')
