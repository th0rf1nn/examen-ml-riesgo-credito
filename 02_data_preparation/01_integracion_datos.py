import pandas as pd
import os

# --- CONFIGURACIÓN ROBUSTA (A prueba de errores de ruta) ---
# 1. Obtenemos la ruta exacta de DONDE está este archivo script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construimos la ruta a la carpeta data (subimos un nivel desde el script y entramos a data)
# Esto funcionará sin importar desde qué terminal lo ejecutes.
DATA_DIR = os.path.join(script_dir, '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'dataset_integrado.parquet')

# Nombres de archivos
FILE_APP = 'application_.parquet'
FILE_BUREAU = 'bureau.parquet'

def cargar_datos():
    print(f"1. Buscando datos en: {os.path.abspath(DATA_DIR)}")
    
    path_app = os.path.join(DATA_DIR, FILE_APP)
    path_bureau = os.path.join(DATA_DIR, FILE_BUREAU)
    
    # Verificación explicita
    if not os.path.exists(path_app): 
        raise FileNotFoundError(f"❌ NO ENCUENTRO: {path_app}")
    
    print("Archivos encontrados. Cargando...")
    df_app = pd.read_parquet(path_app, engine='pyarrow')
    df_bureau = pd.read_parquet(path_bureau, engine='pyarrow')
    
    return df_app, df_bureau

def crear_features_bureau(bureau_df):
    """Genera ingeniería de características agregadas."""
    print("2. Procesando Bureau (Feature Engineering)...")
    
    aggregations = {
        'DAYS_CREDIT': ['mean', 'min', 'max'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum']
    }
    
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(aggregations)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    
    bureau_count = bureau_df.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_LOAN_COUNT')
    bureau_final = bureau_agg.merge(bureau_count, on='SK_ID_CURR', how='left')
    
    return bureau_final

def main():
    try:
        # 1. Cargar
        df_app, df_bureau = cargar_datos()
        
        # 2. Ingeniería
        df_bureau_agg = crear_features_bureau(df_bureau)
        
        # 3. Unir
        print("3. Uniendo tablas...")
        df_final = df_app.merge(df_bureau_agg, on='SK_ID_CURR', how='left')
        
        # 4. Guardar
        print(f"4. Guardando en: {OUTPUT_FILE}")
        df_final.to_parquet(OUTPUT_FILE, engine='pyarrow')
        
        print("\n" + "="*40)
        print("INTEGRACIÓN REALIZADA CORRECTAMENTE")
        print(f"Columnas Iniciales: {df_app.shape[1]}")
        print(f"Columnas Finales:   {df_final.shape[1]} (Variables agregadas correctamente)")
        print("="*40)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")

if __name__ == "__main__":
    main()