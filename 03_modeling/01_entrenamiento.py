import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

# --- CONFIGURACIÓN ---
# Rutas dinámicas (A prueba de errores)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'dataset_integrado.parquet')
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, '..', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'modelo_xgboost.joblib')

def main():
    print("1. Cargando dataset integrado...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ Falta el archivo: {DATA_FILE}")
        
    df = pd.read_parquet(DATA_FILE, engine='pyarrow')
    
    # 2. Separar Variables (X) y Objetivo (y)
    # TARGET es lo que queremos predecir (1 = No paga, 0 = Paga)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) # Sacamos el ID porque no predice nada
    y = df['TARGET']
    
    # Manejo de columnas categóricas (Texto)
    # XGBoost necesita que todo sea número. Usamos "One-Hot Encoding" rápido
    print("2. Preprocesando datos (Encoding)...")
    X = pd.get_dummies(X, drop_first=True)
    
    # Alineamos las columnas para evitar errores futuros
    feature_names = X.columns.tolist()
    
    # 3. División Train / Test (80% entrenar, 20% validar)
    print("3. Dividiendo set de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 'stratify=y' es CLAVE: asegura que haya la misma proporción de deudores en ambos grupos.
    
    # 4. Configuración del Modelo y Manejo de Desbalance
    print("4. Entrenando modelo XGBoost ")
    
    # Calculamos el ratio para el desbalance
    # (Cantidad de Pagadores / Cantidad de Deudores)
    ratio_desbalance = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    
    model = xgb.XGBClassifier(
        n_estimators=100,       # Número de árboles
        learning_rate=0.1,      # Velocidad de aprendizaje
        max_depth=4,            # Profundidad de los árboles (evita sobreajuste)
        scale_pos_weight=ratio_desbalance, # <--- AQUÍ ESTÁ LA MAGIA DEL DESBALANCE
        random_state=42,
        n_jobs=-1               # Usar todos los núcleos del PC
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluación Rápida (Para saber si funcionó)
    print("\n--- RESULTADOS PRELIMINARES ---")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    print(f"AUC-ROC: {auc:.4f} (Ideal > 0.7)")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, preds))
    
    # 6. Guardar el modelo y los nombres de columnas
    print(f"6. Guardando artefactos en {ARTIFACTS_DIR}...")
    
    # Guardamos el modelo
    joblib.dump(model, MODEL_PATH)
    
    # Guardamos también la lista de columnas (vital para la API después)
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, 'features.joblib'))
    
    print("MODELO ENTRENADO Y GUARDADO")

if __name__ == "__main__":
    main()