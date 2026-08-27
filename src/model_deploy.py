# src/model_deploy.py
# API para despliegue del modelo con FastAPI

import os
import sys
import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

# Agregar el directorio src al path para importar ft_engineering
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ft_engineering import preprocesar_datos, cargar_datos


###############################
# 1. Definición de modelos de datos
###############################

class ClienteInput(BaseModel):
    """Modelo para un solo cliente."""
    tipo_credito: int
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: int
    total_otros_prestamos: int
    cuota_pactada: int
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_credito": 4,
                "fecha_prestamo": "2025-01-15 10:30:00",
                "capital_prestado": 1200000.0,
                "plazo_meses": 12,
                "edad_cliente": 35,
                "tipo_laboral": "Empleado",
                "salario_cliente": 3000000,
                "total_otros_prestamos": 1000000,
                "cuota_pactada": 100000,
                "puntaje": 85.0,
                "puntaje_datacredito": 750.0,
                "cant_creditosvigentes": 3,
                "huella_consulta": 2,
                "saldo_mora": 0.0,
                "saldo_total": 50000.0,
                "saldo_principal": 50000.0,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 1000000.0,
                "tendencia_ingresos": "Creciente"
            }
        }


class BatchInput(BaseModel):
    """Modelo para lote de clientes."""
    clients: List[ClienteInput]


class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicción."""
    prediction: int
    probability: float
    message: str


class BatchPredictionResponse(BaseModel):
    """Modelo de respuesta para predicción en lote."""
    predictions: List[PredictionResponse]
    total: int


###############################
# 2. Inicialización de la API
###############################

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="""
    ## API de Riesgo Crediticio
    
    Esta API permite predecir si un cliente pagará a tiempo su crédito.
    
    ### Endpoints disponibles:
    
    - `GET /saludo` - Verificar que la API está funcionando
    - `GET /health` - Health check
    - `POST /predict` - Predecir para un solo cliente
    - `POST /predict/batch` - Predecir para múltiples clientes
    """,
    version="1.1.1"
)


###############################
# 3. Cargar el modelo y el preprocesador
###############################

modelo = None
preprocessor = None

try:
    # Cargar el modelo entrenado
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.pkl")
    with open(model_path, "rb") as f:
        modelo = pickle.load(f)
    print("Modelo cargado correctamente.")
    
    # Cargar el preprocesador
    # Opción: podemos guardar el preprocesador por separado o usar el que viene con el pipeline
    if hasattr(modelo, 'named_steps') and 'preprocessor' in modelo.named_steps:
        preprocessor = modelo.named_steps['preprocessor']
        print("Preprocesador cargado correctamente.")
    else:
        print("AVISO: El modelo no contiene un preprocesador. Se usará el pipeline completo.")
        
except FileNotFoundError:
    print("ERROR: No se encontró el archivo del modelo. Ejecuta model_training_evaluation.py primero.")
except Exception as e:
    print(f"ERROR al cargar el modelo: {e}")


###############################
# 4. Funciones auxiliares
###############################

def preparar_datos_para_prediccion(cliente: ClienteInput) -> pd.DataFrame:
    """Convierte un ClienteInput a DataFrame para predicción."""
    data = cliente.model_dump() if hasattr(cliente, "model_dump") else cliente.dict()
    # Convertir fecha a datetime
    data['fecha_prestamo'] = pd.to_datetime(data['fecha_prestamo'])
    return pd.DataFrame([data])


###############################
# 5. Endpoints de la API
###############################

@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "API de Predicción de Pago a Tiempo",
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }


@app.get("/saludo")
async def saludo():
    """Endpoint de saludo para verificar que la API está funcionando."""
    return {
        "mensaje": "Hola! Esta API sirve para predecir si un cliente pagará a tiempo o no.",
        "status": "running desde el contenedor"
    }


@app.get("/health")
async def health_check():
    """Health check para verificar el estado de la API."""
    return {
        "status": "healthy",
        "model_loaded": modelo is not None,
        "version": "1.1.1"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(cliente: ClienteInput):
    """
    Predice si un cliente pagará a tiempo.
    
    - **prediction**: 1 = Pagará a tiempo, 0 = No pagará
    - **probability**: Probabilidad de pago a tiempo
    """
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Contacte al administrador."
        )
    
    try:
        # Preparar datos
        df = preparar_datos_para_prediccion(cliente)
        
        # Hacer predicción
        prediction = modelo.predict(df)[0]
        probability = modelo.predict_proba(df)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            message="Cliente evaluado exitosamente"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la solicitud: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchInput):
    """
    Predice para múltiples clientes en lote.
    """
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Contacte al administrador."
        )
    
    try:
        # Preparar datos
        data_list = [
            cliente.model_dump() if hasattr(cliente, "model_dump") else cliente.dict()
            for cliente in batch.clients
        ]
        df = pd.DataFrame(data_list)
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
        
        # Hacer predicciones
        predictions = modelo.predict(df)
        probabilities = modelo.predict_proba(df)[:, 1]
        
        # Crear respuestas
        responses = []
        for pred, prob in zip(predictions, probabilities):
            responses.append(PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                message="Cliente evaluado exitosamente"
            ))
        
        return BatchPredictionResponse(
            predictions=responses,
            total=len(responses)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la solicitud: {str(e)}")


###############################
# 6. Ejecutar la API
###############################

if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
