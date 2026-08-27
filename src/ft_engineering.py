# src/ft_engineering.py
# Funciones de preprocesamiento y feature engineering

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def convertir_categoricas_a_texto(X):
    """Normaliza categorías mixtas sin convertir los valores ausentes."""
    return X.apply(lambda col: col.map(lambda value: str(value) if pd.notna(value) else np.nan))


def cargar_datos():
    """
    Carga los datos desde el archivo Excel.
    """
    # Ruta absoluta del directorio donde está este archivo (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Subir un nivel para llegar a la carpeta donde está la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)
    
    # Construir la ruta completa al Excel
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")
    
    # Leer los datos
    df = pd.read_excel(ruta_excel)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
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
    
    # 2. Eliminar columnas que no se usan para entrenar
    columns_to_drop = ['fecha_prestamo']
    X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
    
    # 3. Identificar tipos de columnas
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    
    print(f"Features: {X.shape[1]} columnas")
    print(f"Target: {y.name}")
    print(f"\nColumnas numéricas ({len(numerical_cols)}): {len(numerical_cols)}")
    print(f"Columnas categóricas ({len(categorical_cols)}): {len(categorical_cols)}")
    
    # 4. Crear transformadores para preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('to_string', FunctionTransformer(convertir_categoricas_a_texto)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 5. Combinar en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # 6. Crear pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    # 7. Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nDivisión de datos:")
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test: {X_test.shape[0]} muestras")
    
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


if __name__ == "__main__":
    df = cargar_datos()
    data_dict = preprocesar_datos(df)
    print("\nFeature Engineering completado!")
