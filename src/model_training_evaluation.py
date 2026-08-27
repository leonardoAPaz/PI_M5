# src/model_training_evaluation.py
# Entrenamiento, evaluación y guardado del modelo

import os
import sys
from math import prod
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ft_engineering import cargar_datos, preprocesar_datos


def entrenar_modelo(X_train, y_train, preprocessor):
    """
    Entrena y optimiza un modelo Random Forest con GridSearch.
    """
    print("\n" + "="*50)
    print("ENTRENAMIENTO DEL MODELO")
    print("="*50 + "\n")
    
    # Definir pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Grid de hiperparámetros
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    print("Realizando búsqueda de hiperparámetros (GridSearchCV)...")
    print(f"   {prod(len(values) for values in param_grid.values())} combinaciones a evaluar\n")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nMejores parámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\nMejor puntaje (CV ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def evaluar_modelo(model, X_test, y_test):
    """
    Evalúa el modelo y retorna métricas.
    """
    print("\n" + "="*50)
    print("EVALUACIÓN DEL MODELO")
    print("="*50 + "\n")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Métricas de evaluación:")
    for key, value in metrics.items():
        print(f"   {key.upper()}: {value:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(pd.DataFrame(cm, columns=['Pred: 0', 'Pred: 1'], index=['Real: 0', 'Real: 1']))
    
    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No paga', 'Paga a tiempo']))
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC del Modelo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    plt.savefig(os.path.join(models_dir, 'roc_curve.png'))
    plt.close()
    
    return metrics, y_pred, y_pred_proba


def guardar_modelo(model, filename='model.pkl'):
    """
    Guarda el modelo entrenado en la carpeta models.
    """
    # Asegurar que la carpeta models existe
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    filepath = os.path.join(models_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModelo guardado en: {filepath}")
    return filepath


def main():
    """
    Ejecuta todo el pipeline de entrenamiento.
    """
    print("INICIANDO PIPELINE DE ENTRENAMIENTO\n")
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Preprocesar
    data_dict = preprocesar_datos(df)
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    preprocessor = data_dict['preprocessor']
    
    # 3. Entrenar modelo
    model, best_params = entrenar_modelo(X_train, y_train, preprocessor)
    
    # 4. Evaluar modelo
    metrics, _, _ = evaluar_modelo(model, X_test, y_test)
    
    # 5. Guardar modelo
    model_path = guardar_modelo(model)
    
    # 6. Resumen final
    print("\n" + "="*50)
    print("PIPELINE DE ENTRENAMIENTO COMPLETADO")
    print("="*50)
    print("\nResumen del modelo:")
    print(f"   Modelo: RandomForestClassifier")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   Modelo guardado en: {model_path}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
