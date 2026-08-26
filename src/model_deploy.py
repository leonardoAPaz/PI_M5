# src/model_deploy.py

# librerías
import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

###############################
### 1. Inicialización de la API
###############################

app = FastAPI(title="API de predicción de Pago a Tiempo",
              description="Esta API permite predecir si un cliente pagará a tiempo o no.",
              version="1.1.1"
              )

###################################
### 2. Cargamos el modelo entrenado
###################################

try:
    # 2.1 Aqui cargamos el modelo entrenado
    with open("../models/model.pkl", "rb") as f:
        modelo = pickle.load(f)

    print("Modelo cargado correctamente.")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

########################################
### 3. Definimos los endpoints de la API
########################################

# 3.1 Endpoint de saludo
@app.get("/saludo")
def saludo():
    return {"mensaje":"Hola! Esta api sirve para predecir si un cliente pagará a tiempo o no. Además, estoy corriendo desde el contendor."}

# 3.2 Endpoint de predicción
@app.post("/predict")
def predict_batch(input_data:dict):
    if modelo is None:
        return {"El modelo no pudo ser caragdo. Revisa los logs del servidor para más detalles."}

    try:
        return {"El modelo está cargado y listo para hacer predicciones."}

    except Exception as e:
        return {f"Error al hacer la predicción: {e}"}