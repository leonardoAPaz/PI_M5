import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def cargar_datos():
    """
    Carga los datos desde el archivo Excel.
    """
    # 1. Ruta absoluta del directorio donde está este archivo (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # 2. Subir un nivel para llegar a la carpeta donde está la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)

    # 3. Construir la ruta completa al Excel
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")

    # 4. Leemos los datos
    df = pd.read_excel(ruta_excel)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def explorar_datos(df):
    """
    Realiza un análisis exploratorio básico de los datos.
    """
    print("\n" + "="*50)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*50 + "\n")

    print("🔍 Primeras 5 filas:")
    print(df.head(), "\n")

    print("📊 Información general del DataFrame:")
    print(df.info(), "\n")

    print("📈 Estadísticas descriptivas de variables numéricas:")
    print(df.describe(), "\n")

    print("📋 Tipos de datos por columna:")
    print(df.dtypes, "\n")

    # Identificar valores nulos
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print("⚠️  Columnas con valores nulos:")
        print(nulos[nulos > 0], "\n")
    else:
        print("✅ No hay valores nulos en el dataset.\n")

    # Identificar columnas con valores que parecen outliers extremos
    # (como valores que son strings pero deberían ser números)
    print("🔎 Revisando columnas con valores sospechosos:")
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            # Si tiene muchos valores únicos y parece texto libre, puede ser problemático
            if len(unique_vals) > 50:
                print(f"   - Columna '{col}': {len(unique_vals)} valores únicos. ¿Es texto libre?")
        elif df[col].dtype in ['int64', 'float64']:
            # Buscar valores extremadamente grandes o pequeños
            if df[col].max() > 1e12 or df[col].min() < -1e12:
                print(f"   - Columna '{col}': valores extremos (min: {df[col].min():.2e}, max: {df[col].max():.2e})")

    # Revisar la variable objetivo 'Pago_atiempo'
    print(f"\n🎯 Distribución de la variable objetivo 'Pago_atiempo':")
    print(df['Pago_atiempo'].value_counts(normalize=True) * 100)

    # Correlación con la variable objetivo (si es numérica)
    print("\n📊 Correlación de variables numéricas con 'Pago_atiempo':")
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Filtrar la variable objetivo y columnas que podrían ser ID o fecha
    cols_to_check = [col for col in numeric_cols if col != 'Pago_atiempo' and col not in ['tipo_credito']]
    if cols_to_check:
        correlations = df[cols_to_check + ['Pago_atiempo']].corr()['Pago_atiempo'].sort_values(ascending=False)
        print(correlations)

    return df


def preprocesar_datos(df, target='Pago_atiempo'):
    """
    Prepara los datos para el modelado:
    - Separa features y target
    - Identifica columnas numéricas y categóricas
    - Crea pipelines de preprocesamiento
    """
    print("\n" + "="*50)
    print("PREPROCESAMIENTO DE DATOS")
    print("="*50 + "\n")

    # 1. Separar features (X) y target (y)
    X = df.drop(target, axis=1)
    y = df[target]

    print(f"✅ Features: {X.shape[1]} columnas")
    print(f"✅ Target: {y.name}")

    # 2. Identificar tipos de columnas
    # - Columnas que son claramente IDs o fechas (no se usan para entrenar)
    columns_to_drop = ['fecha_prestamo']  # La fecha no es útil directamente

    # - Columnas numéricas
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Quitar columnas que no queremos usar (IDs, fechas, etc.)
    numerical_cols = [col for col in numerical_cols if col not in columns_to_drop]

    # - Columnas categóricas (object o string)
    categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in columns_to_drop]

    # - Columnas que podrían ser IDs (si tienen muchos valores únicos)
    #   Se recomienda revisar manualmente, pero podemos hacer una aproximación
    potential_id_cols = []
    for col in categorical_cols:
        if X[col].nunique() > 100:  # Si tiene más de 100 valores únicos, probablemente es un ID
            potential_id_cols.append(col)

    # Quitamos las IDs de las categóricas
    for col in potential_id_cols:
        if col in categorical_cols:
            categorical_cols.remove(col)

    print(f"\n📋 Columnas numéricas ({len(numerical_cols)}):")
    print(f"   {numerical_cols}")

    print(f"\n📋 Columnas categóricas ({len(categorical_cols)}):")
    print(f"   {categorical_cols}")

    if potential_id_cols:
        print(f"\n⚠️  Posibles columnas ID (no se usarán): {potential_id_cols}")

    # 3. Crear transformadores para preprocesamiento
    # Para numéricas: imputar con mediana y escalar
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Para categóricas: imputar con moda y one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. Combinar en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Las columnas que no se usan se descartan
    )

    # 5. Crear un pipeline completo (por ahora solo el preprocesamiento)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    print("\n✅ Pipeline de preprocesamiento creado exitosamente.")

    # 6. Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 División de datos:")
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test: {X_test.shape[0]} muestras")

    # 7. Aplicar el preprocesamiento (opcional, para ver las dimensiones)
    # Nota: Esto transforma los datos, pero no es necesario hacerlo aquí
    # si lo vamos a hacer en el pipeline de entrenamiento

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'pipeline': pipeline,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'X_full': X,
        'y_full': y
    }


def main():
    """
    Función principal para ejecutar todo el feature engineering.
    """
    # Cargar datos
    df = cargar_datos()

    # Explorar datos
    df = explorar_datos(df)

    # Preprocesar datos
    data_dict = preprocesar_datos(df)

    print("\n✅ Feature Engineering completado!")

    return df, data_dict


if __name__ == "__main__":
    # Si se ejecuta este script directamente, corre todo el proceso
    df, data_dict = main()

    # Guardar los objetos principales en variables para usarlos después
    # (en el notebook o en el siguiente script)
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    preprocessor = data_dict['preprocessor']
    pipeline = data_dict['pipeline']

    print("\n" + "="*50)
    print("✅ ¡Todo listo para el modelado!")
    print("="*50)