# src/model_monitoring.py
# Dashboard de monitoreo con Streamlit

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, ks_2samp

# Agregar el directorio src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ft_engineering import cargar_datos

###############################
# 1. Configuración
###############################

st.set_page_config(
    page_title="Monitoreo de Modelo - Riesgo Crediticio",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Base_de_datos.xlsx")


###############################
# 2. Funciones de carga de datos
###############################

@st.cache_data
def load_and_split_data():
    """Carga los datos y los divide en referencia y nuevos."""
    df = cargar_datos()
    
    # Ordenar por fecha para simular datos históricos vs nuevos
    df = df.sort_values('fecha_prestamo')
    split_idx = int(len(df) * 0.7)
    
    df_ref = df.iloc[:split_idx]  # Datos históricos (referencia)
    df_new = df.iloc[split_idx:]   # Datos nuevos
    
    return df_ref, df_new, df


@st.cache_data
def get_drift_metrics(df_ref, df_new, numeric_cols, categorical_cols):
    """Calcula métricas de drift para variables numéricas y categóricas."""
    results = {}
    
    # Drift numérico (KS test)
    for col in numeric_cols:
        if col in df_ref.columns and col in df_new.columns:
            stat, p_value = ks_2samp(df_ref[col].dropna(), df_new[col].dropna())
            results[col] = {
                'type': 'numeric',
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05,
                'mean_ref': df_ref[col].mean(),
                'mean_new': df_new[col].mean()
            }
    
    # Drift categórico (Chi-cuadrado)
    for col in categorical_cols:
        if col in df_ref.columns and col in df_new.columns:
            # Obtener categorías comunes
            categories = pd.concat([df_ref[col], df_new[col]]).unique()
            ref_counts = df_ref[col].value_counts().reindex(categories, fill_value=0)
            new_counts = df_new[col].value_counts().reindex(categories, fill_value=0)
            contingency = pd.DataFrame([ref_counts, new_counts])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            results[col] = {
                'type': 'categorical',
                'chi2': chi2,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            }
    
    return results


###############################
# 3. Interfaz Principal
###############################

st.title("📊 Aplicación de Monitoreo del Modelo")
st.markdown("### Riesgo Crediticio - Predicción de Pago a Tiempo")

# Barra lateral
st.sidebar.header("🔍 Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    ["📈 Visión General", "🔬 Análisis de Drift", "🧪 Probar Modelo", "📊 Estadísticas"]
)

# Cargar datos
with st.spinner("Cargando datos..."):
    df_ref, df_new, df_full = load_and_split_data()

# Identificar columnas
numeric_cols = df_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_full.select_dtypes(include=['object', 'str']).columns.tolist()


###############################
# SECCIÓN: Visión General
###############################

if section == "📈 Visión General":
    st.header("📈 Visión General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", f"{len(df_full):,}")
    with col2:
        st.metric("Columnas", len(df_full.columns))
    with col3:
        st.metric("Datos Referencia", f"{len(df_ref):,}")
    with col4:
        st.metric("Datos Nuevos", f"{len(df_new):,}")
    
    st.subheader("Distribución de 'Pago_atiempo'")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        df_full['Pago_atiempo'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Conteo por Clase')
        ax.set_xlabel('Pago a tiempo (1=Sí, 0=No)')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        df_full['Pago_atiempo'].value_counts(normalize=True).plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Proporción por Clase')
        st.pyplot(fig)
    
    st.info(f"**Nota:** El {df_full['Pago_atiempo'].mean()*100:.1f}% de los clientes pagan a tiempo. El dataset está desbalanceado.")

###############################
# SECCIÓN: Análisis de Drift
###############################

elif section == "🔬 Análisis de Drift":
    st.header("🔬 Análisis de Data Drift")
    
    st.write("""
    El data drift ocurre cuando la distribución de los datos cambia con el tiempo.
    Esto puede afectar el rendimiento del modelo.
    """)
    
    # Calcular drift
    with st.spinner("Calculando métricas de drift..."):
        drift_results = get_drift_metrics(df_ref, df_new, numeric_cols, categorical_cols)
    
    # Resumen
    total_vars = len(drift_results)
    drift_detected = sum(1 for v in drift_results.values() if v.get('drift_detected', False))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variables Analizadas", total_vars)
    with col2:
        st.metric("Drift Detectado", drift_detected)
    with col3:
        pct = (drift_detected / total_vars * 100) if total_vars > 0 else 0
        st.metric("Porcentaje con Drift", f"{pct:.1f}%")
    
    # Tabla de resultados
    st.subheader("Resultados por Variable")
    
    results_data = []
    for var, metrics in drift_results.items():
        row = {
            'Variable': var,
            'Tipo': metrics['type'],
            'Drift Detectado': '⚠️ Sí' if metrics.get('drift_detected', False) else '✅ No'
        }
        if metrics['type'] == 'numeric':
            row['Estadístico'] = f"{metrics.get('statistic', 0):.4f}"
            row['p-value'] = f"{metrics.get('p_value', 0):.4f}"
            row['Cambio Media (%)'] = f"{abs(metrics.get('mean_new', 0) - metrics.get('mean_ref', 0)) / (metrics.get('mean_ref', 1) + 1e-10) * 100:.1f}%"
        else:
            row['Chi2'] = f"{metrics.get('chi2', 0):.2f}"
            row['p-value'] = f"{metrics.get('p_value', 0):.4f}"
            row['Cambio Media (%)'] = "-"
        results_data.append(row)
    
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Visualización de drift en variables numéricas
    st.subheader("Distribución de Variables Numéricas (Referencia vs Nuevos)")
    
    numeric_with_drift = [v for v, m in drift_results.items() 
                         if m.get('type') == 'numeric' and m.get('drift_detected', False)]
    
    if numeric_with_drift:
        cols_to_plot = numeric_with_drift[:6]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            if i < len(axes):
                sns.kdeplot(df_ref[col].dropna(), label='Referencia', ax=axes[i])
                sns.kdeplot(df_new[col].dropna(), label='Nuevos', ax=axes[i])
                axes[i].set_title(f'{col}')
                axes[i].legend()
        
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.success("✅ No se detectó drift significativo en variables numéricas.")
    
    # Variables categóricas con drift
    st.subheader("Variables Categóricas con Drift")
    cat_with_drift = [v for v, m in drift_results.items() 
                     if m.get('type') == 'categorical' and m.get('drift_detected', False)]
    
    if cat_with_drift:
        for col in cat_with_drift:
            st.write(f"**{col}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Distribución Referencia:")
                st.write(df_ref[col].value_counts(normalize=True).head())
            with col2:
                st.write("Distribución Nuevos:")
                st.write(df_new[col].value_counts(normalize=True).head())
            st.divider()
    else:
        st.success("✅ No se detectó drift en variables categóricas.")
    
    # Recomendaciones
    st.subheader("💡 Recomendaciones")
    if drift_detected > 0:
        st.warning(f"""
        ⚠️ Se detectó drift en {drift_detected} variables.
        
        **Acciones recomendadas:**
        1. Monitorear el rendimiento del modelo en producción
        2. Considerar reentrenar el modelo con datos más recientes
        3. Revisar si las variables con drift son importantes para el modelo
        """)
    else:
        st.success("✅ No se detectó drift significativo. El modelo debería mantener buen rendimiento.")

###############################
# SECCIÓN: Probar Modelo
###############################

elif section == "🧪 Probar Modelo":
    st.header("🧪 Probar el Modelo")
    
    st.write("""
    Ingresa los datos de un cliente para obtener una predicción.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        tipo_credito = st.selectbox("Tipo de Crédito", sorted(df_full['tipo_credito'].unique()))
        capital_prestado = st.number_input("Capital Prestado", min_value=0, value=1200000)
        plazo_meses = st.slider("Plazo (meses)", 1, 36, 12)
        edad_cliente = st.slider("Edad del Cliente", 18, 80, 35)
        tipo_laboral = st.selectbox("Tipo Laboral", df_full['tipo_laboral'].unique())
        salario_cliente = st.number_input("Salario del Cliente", min_value=0, value=3000000)
        total_otros_prestamos = st.number_input("Total Otros Préstamos", min_value=0, value=1000000)
    
    with col2:
        cuota_pactada = st.number_input("Cuota Pactada", min_value=0, value=100000)
        puntaje = st.slider("Puntaje", 0.0, 100.0, 85.0)
        puntaje_datacredito = st.number_input("Puntaje DataCrédito", min_value=0.0, value=750.0)
        cant_creditosvigentes = st.number_input("Créditos Vigentes", min_value=0, value=3)
        huella_consulta = st.number_input("Huella de Consulta", min_value=0, value=2)
        saldo_mora = st.number_input("Saldo en Mora", min_value=0.0, value=0.0)
        saldo_total = st.number_input("Saldo Total", min_value=0.0, value=50000.0)
        saldo_principal = st.number_input("Saldo Principal", min_value=0.0, value=50000.0)
        saldo_mora_codeudor = st.number_input("Saldo Mora Codeudor", min_value=0.0, value=0.0)
        creditos_sectorFinanciero = st.number_input("Créditos Sector Financiero", min_value=0, value=1)
        creditos_sectorCooperativo = st.number_input("Créditos Sector Cooperativo", min_value=0, value=0)
        creditos_sectorReal = st.number_input("Créditos Sector Real", min_value=0, value=1)
        promedio_ingresos_datacredito = st.number_input("Promedio Ingresos DataCrédito", min_value=0.0, value=1000000.0)
        tendencia_ingresos = st.selectbox("Tendencia Ingresos", df_full['tendencia_ingresos'].dropna().unique())
    
    if st.button("🔮 Predecir", type="primary"):
        # Verificar que la API esté corriendo
        try:
            # Construir payload
            payload = {
                "tipo_credito": int(tipo_credito),
                "fecha_prestamo": "2025-01-15 10:30:00",
                "capital_prestado": float(capital_prestado),
                "plazo_meses": int(plazo_meses),
                "edad_cliente": int(edad_cliente),
                "tipo_laboral": tipo_laboral,
                "salario_cliente": int(salario_cliente),
                "total_otros_prestamos": int(total_otros_prestamos),
                "cuota_pactada": int(cuota_pactada),
                "puntaje": float(puntaje),
                "puntaje_datacredito": float(puntaje_datacredito),
                "cant_creditosvigentes": int(cant_creditosvigentes),
                "huella_consulta": int(huella_consulta),
                "saldo_mora": float(saldo_mora),
                "saldo_total": float(saldo_total),
                "saldo_principal": float(saldo_principal),
                "saldo_mora_codeudor": float(saldo_mora_codeudor),
                "creditos_sectorFinanciero": int(creditos_sectorFinanciero),
                "creditos_sectorCooperativo": int(creditos_sectorCooperativo),
                "creditos_sectorReal": int(creditos_sectorReal),
                "promedio_ingresos_datacredito": float(promedio_ingresos_datacredito),
                "tendencia_ingresos": tendencia_ingresos
            }
            
            response = requests.post(f"{API_URL}", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Predicción exitosa")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicción", "Paga a tiempo" if result['prediction'] == 1 else "No paga")
                with col2:
                    st.metric("Probabilidad", f"{result['probability']*100:.1f}%")
                
                st.info(f"📝 {result['message']}")
            else:
                st.error(f"❌ Error en la API: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar a la API. Asegúrate de que FastAPI esté corriendo.")
            st.code("uvicorn src.model_deploy:app --reload")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

###############################
# SECCIÓN: Estadísticas
###############################

else:
    st.header("📊 Estadísticas del Modelo")
    
    st.subheader("Distribución de Variables")
    
    # Selección de variable
    var_selected = st.selectbox("Selecciona una variable para visualizar:", df_full.columns)
    
    if var_selected:
        col1, col2 = st.columns(2)
        
        with col1:
            if df_full[var_selected].dtype in ['int64', 'float64']:
                fig, ax = plt.subplots()
                sns.histplot(df_full[var_selected], kde=True, ax=ax)
                ax.set_title(f'Distribución de {var_selected}')
                st.pyplot(fig)
                
                # Estadísticas
                st.write("**Estadísticas:**")
                st.write(df_full[var_selected].describe())
            else:
                fig, ax = plt.subplots()
                df_full[var_selected].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Distribución de {var_selected}')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                
                st.write("**Frecuencias:**")
                st.write(df_full[var_selected].value_counts())
        
        with col2:
            # Relación con el target
            fig, ax = plt.subplots()
            if df_full[var_selected].dtype in ['int64', 'float64']:
                for label in [0, 1]:
                    data = df_full[df_full['Pago_atiempo'] == label][var_selected]
                    sns.kdeplot(data, label=f'Pago={label}', ax=ax)
                ax.set_title(f'{var_selected} vs Pago_atiempo')
                ax.legend()
            else:
                cross = pd.crosstab(df_full[var_selected], df_full['Pago_atiempo'], normalize='index')
                cross.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f'{var_selected} vs Pago_atiempo')
                ax.legend(title='Pago', labels=['No', 'Sí'])
            st.pyplot(fig)
    
    # Correlaciones
    st.subheader("Correlaciones con Pago_atiempo")
    numeric_cols_list = df_full.select_dtypes(include=['int64', 'float64']).columns
    corr = df_full[numeric_cols_list].corr()['Pago_atiempo'].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    corr[1:11].plot(kind='barh', ax=ax)
    ax.set_title('Top 10 Variables Correlacionadas con Pago_atiempo')
    ax.set_xlabel('Correlación')
    st.pyplot(fig)