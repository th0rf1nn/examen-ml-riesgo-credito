import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

# --- CONFIGURACIÓN ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'dataset_integrado.parquet')
MODEL_FILE = os.path.join(SCRIPT_DIR, '..', 'artifacts', 'modelo_xgboost.joblib')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'artifacts') # Guardaremos las fotos aquí

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión (Predicción de Riesgo)', fontsize=14)
    plt.xlabel('Predicción (0=Paga, 1=No Paga)')
    plt.ylabel('Realidad')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Matriz de confusión guardada en: {output_path}")

def plot_roc_curve(y_true, y_probs, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title('Curva ROC - Rendimiento del Modelo', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Curva ROC guardada en: {output_path}")

def plot_feature_importance(model, feature_names, output_path):
    # Extraer importancia
    importance = model.feature_importances_
    # Crear DataFrame para ordenar
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df_imp = df_imp.sort_values('importance', ascending=False).head(20) # Top 20
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_imp, palette='viridis')
    plt.title('Top 20 Variables más Importantes para el Modelo', fontsize=14)
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Importancia de características guardada en: {output_path}")

def main():
    print("1. Cargando datos y modelo...")
    df = pd.read_parquet(DATA_FILE, engine='pyarrow')
    model = joblib.load(MODEL_FILE)
    
    # Preprocesamiento (igual que en el entrenamiento)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    X = pd.get_dummies(X, drop_first=True)
    
    # Recreamos el split exacto usando la misma semilla (42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("2. Generando predicciones...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    print("3. Creando visualizaciones...")
    plot_confusion_matrix(y_test, y_pred, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_probs, os.path.join(OUTPUT_DIR, 'roc_curve.png'))
    
    # Para importancia de características, necesitamos los nombres de las columnas
    plot_feature_importance(model, X.columns, os.path.join(OUTPUT_DIR, 'feature_importance.png'))

if __name__ == "__main__":
    main()