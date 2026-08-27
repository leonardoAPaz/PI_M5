# 🏦 Modelo de Riesgo Crediticio - Proyecto Integrador M5

## 📌 Descripción del Proyecto

Este proyecto desarrolla un modelo predictivo de riesgo crediticio para una empresa financiera, utilizando técnicas de aprendizaje automático. El modelo estima la probabilidad de que un cliente pague o no su crédito a tiempo, permitiendo tomar decisiones más informadas en la asignación de préstamos.

El proyecto sigue una **arquitectura MLOps** con:
- Versionamiento de código (Git/GitHub)
- Pipeline de datos automatizado
- Entrenamiento y evaluación de modelos
- API para despliegue (FastAPI)
- Dashboard de monitoreo (Streamlit)
- Contenerización (Docker)

---

## 📁 Estructura del Proyecto

```text
PI_M5/
├── models/ # Modelos entrenados (.pkl)
│ ├── model.pkl # Modelo final
│ └── roc_curve.png # Curva ROC del modelo
│
├── src/ # Código fuente principal
│ ├── cargar_datos.py # Función de carga de datos
│ ├── ft_engineering.py # Feature engineering y preprocesamiento
│ ├── model_deploy.py # API FastAPI para despliegue
│ ├── model_monitoring.py # Dashboard Streamlit para monitoreo
│ └── model_training_evaluation.py # Entrenamiento y evaluación del modelo
│ └── comprension_eda.ipynb # Jupyter notebooks para EDA (Análisis Exploratorio de Datos)
│
├── .gitignore # Archivos ignorados por Git
├── Base_de_datos.xlsx # Dataset original
├── Base_de_datos_con_Data_Drift_Simulado.xlsx # Dataset con drift simulado
├── Dockerfile # Configuración de Docker
├── LICENSE # Licencia del proyecto
├── README.md # Documentación del proyecto
└── requirements.txt # Dependencias del proyecto
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/leonardoAPaz/PI_M5.git
cd PI_M5
```
### 2. Crear y Activar Entorno Virtual
```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### En Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Pipeline de Datos
1. Carga de Datos
El archivo Base_de_datos.xlsx contiene la información histórica de créditos con 23 columnas y más de 10,000 registros.

2. Análisis Exploratorio (EDA)
Ejecuta el notebook para visualizar el análisis completo:

```bash
jupyter notebook src/comprension_eda.ipynb
```

---

## Hallazgos clave:

Target desbalanceado: 95.3% paga a tiempo, 4.7% no paga

Variables más relevantes: puntaje (correlación 0.92), puntaje_datacredito, edad_cliente

Valores nulos: Especialmente en promedio_ingresos_datacredito (2,930 nulos)

Data drift: Detectado al comparar datos históricos vs nuevos

### 📋 Preprocesamiento

El pipeline de preprocesamiento incluye:

Imputación: Mediana para numéricas, moda para categóricas

Escalado: StandardScaler para variables numéricas

Encoding: One-Hot Encoding para variables categóricas

División: 80% entrenamiento, 20% prueba (estratificado)

### 🤖 Entrenamiento del Modelo

Ejecutar Entrenamiento
```bash
python src/model_training_evaluation.py
```
Modelo Seleccionado

Random Forest Classifier con optimización de hiperparámetros mediante GridSearchCV.

### Hiperparámetros finales:

n_estimators: 100

max_depth: None

min_samples_split: 2

class_weight: 'balanced'

### Métricas de Rendimiento

Métrica	Valor
Accuracy	1.000
Precision	1.000
Recall	1.000
F1-Score	1.000
ROC-AUC	1.000

## 🌐 Despliegue con FastAPI

1. Iniciar la API Localmente
```bash
uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000 --reload
```
2. Endpoints Disponibles

```text
Método	Endpoint	    Descripción
GET	    /               Información de la API
GET	    /saludo	        Verificar estado
GET	    /health	        Health check
POST	/predict	    Predecir un cliente
POST	/predict/batch	Predecir múltiples clientes
```

3. Ejemplo de Solicitud

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 🐳 Despliegue con Docker

1. Construir la Imagen

```bash
docker build -t modelo-riesgo-crediticio .
```
2. Ejecutar el Contenedor

```bash
docker run -p 8000:8000 modelo-riesgo-crediticio
```
3. Verificar

```bash
curl http://localhost:8000/health
```

## 📊 Monitoreo con Streamlit

Iniciar el Dashboard

```bash
streamlit run src/model_monitoring.py
```

Funcionalidades

Visión General: Estadísticas del dataset y distribución del target

Análisis de Drift: Detección de drift en variables numéricas y categóricas

Probar Modelo: Interfaz para probar predicciones

Estadísticas: Visualización detallada de variables

## 📋 Requisitos del Sistema

Se recomienda Python 3.14. Ver requirements.txt para la lista completa de dependencias reproducibles.

```txt
pandas==3.0.3
numpy==2.4.6
matplotlib==3.10.9
seaborn==0.13.2
scikit-learn==1.9.0
scipy==1.18.0
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.4
streamlit==1.62.0
openpyxl==3.1.5
requests==2.34.2
```

## 🧪 Pruebas y Validación

Validación Cruzada
El modelo fue validado usando 5-fold cross-validation con un ROC-AUC promedio de 1.000 en la ejecución reproducida.

Data Drift
Se detectó drift en las siguientes variables al comparar datos históricos vs nuevos:

puntaje_datacredito

capital_prestado

huella_consulta

Recomendación: Monitorear continuamente el rendimiento del modelo y reentrenar periódicamente.

## 👥 Contribuciones
Este proyecto fue desarrollado como parte del Proyecto Integrador del Módulo 5 del programa de Data Science.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
