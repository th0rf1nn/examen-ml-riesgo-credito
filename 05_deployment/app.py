from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Usamos rutas absolutas para no tener errores de "File not found"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, '..', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'modelo_xgboost.joblib')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'features.joblib')

# --- 2. INICIALIZAR LA APP ---
app = FastAPI(
    title="API de Riesgo Crediticio - Examen ML",
    description="Endpoint para predecir Default usando XGBoost",
    version="1.0"
)

# --- 3. CARGAR EL MODELO AL INICIO ---
print(f"Cargando artefactos desde: {ARTIFACTS_DIR}")
model = None
model_features = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        print("✅ Modelo y Features cargados correctamente.")
    else:
        print("❌ ERROR: No encuentro los archivos .joblib en artifacts.")
except Exception as e:
    print(f"❌ Error fatal: {e}")

# --- 4. DEFINIR EL FORMATO DE ENTRADA (JSON) ---
class ClientData(BaseModel):
    # Variables mínimas para probar (puedes agregar más)
    EXT_SOURCE_3: float = 0.5
    EXT_SOURCE_2: float = 0.5
    DAYS_BIRTH: int = -15000
    DAYS_EMPLOYED: int = -2000
    AMT_CREDIT: float = 100000.0
    AMT_INCOME_TOTAL: float = 50000.0
    # Permite recibir cualquier otra columna extra que el modelo necesite
    class Config:
        extra = "allow"

# --- 5. ENDPOINT DE PREDICCIÓN ---
@app.post("/evaluate_risk")
def predict(client: ClientData):
    global model, model_features
    
    if not model:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    try:
        # A. Convertir JSON a DataFrame
        input_data = client.dict()
        df = pd.DataFrame([input_data])
        
        # B. Preprocesamiento (One-Hot Encoding)
        # Convertimos texto a números igual que en el entrenamiento
        df = pd.get_dummies(df)
        
        # C. ALINEACIÓN CRÍTICA (Reindex)
        # Forzamos a que el DataFrame de entrada tenga EXACTAMENTE las mismas columnas
        # que el modelo aprendió. Si falta una, la rellena con 0.
        df_final = df.reindex(columns=model_features, fill_value=0)
        
        # D. Predicción
        # Class 1 = Riesgo de Impago
        probabilidad = float(model.predict_proba(df_final)[0][1])
        clase = int(model.predict(df_final)[0])
        
        # E. Regla de Negocio (Umbral)
        # Si la probabilidad es mayor a 0.5, rechazamos.
        decision = "RECHAZAR" if clase == 1 else "APROBAR"
        
        return {
            "decision": decision,
            "probabilidad_riesgo": round(probabilidad, 4),
            "mensaje": "Alto riesgo detectado" if clase == 1 else "Cliente confiable"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 6. EJECUCIÓN ---
if __name__ == "__main__":
    # Esto permite correrlo con el botón Play de VS Code
    uvicorn.run(app, host="127.0.0.1", port=8000)