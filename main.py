from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# Obtener la ruta del directorio actual del script
script_dir = os.path.dirname(__file__) if __file__ else '.'

# Listar los archivos en el directorio actual
archivos_al_nivel = os.listdir(script_dir)

# Imprimir los archivos al nivel del script
print("Archivos al nivel del script:")
for archivo in archivos_al_nivel:
    print(archivo)


# Cargar el modelo desde el archivo .pkl
with open('model.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)


# Definir la clase de entrada utilizando Pydantic
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

# Crea una instancia de FastAPI
app = FastAPI()

# Define una ruta para hacer predicciones
@app.post("/predecir/")
async def predecir(data: DatosEntrada):
    # Genera una nueva variable con los datos de entrada
    data_dict = {
        "admission_type_id": [data.admission_type_id],
        "discharge_disposition_id": [data.discharge_disposition_id],
        "admission_source_id": [data.admission_source_id],
        "time_in_hospital": [data.time_in_hospital],
        "num_lab_procedures": [data.num_lab_procedures],
        "num_procedures": [data.num_procedures],
        "num_medications": [data.num_medications],
        "number_outpatient": [data.number_outpatient],
        "number_emergency": [data.number_emergency],
        "number_inpatient": [data.number_inpatient],
        "number_diagnoses": [data.number_diagnoses]
    }
    
    # Transforma los datos de entrada en un DataFrame de Pandas
    df = pd.DataFrame(data_dict)
    
    # Realiza la predicción utilizando el modelo
    prediccion = modelo.predict(df)
    # Devuelve la predicción
    return str(prediccion[0])

# Corre el servidor de FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
