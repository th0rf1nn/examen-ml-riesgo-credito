# Predicción de Riesgo de Incumplimiento de Crédito 

## Descripción del Proyecto
Este proyecto desarrolla una solución de Machine Learning completa para predecir la probabilidad de impago (default) de clientes solicitantes de crédito. El objetivo es optimizar la toma de decisiones financieras mediante un modelo automatizado que clasifica a los clientes en:
- **0 (Paga):** Cliente confiable (Aprobar).
- **1 (No Paga):** Cliente de alto riesgo (Rechazar).

##  Estructura del Proyecto (Metodología CRISP-DM)
El repositorio está organizado siguiendo el flujo de trabajo estándar de Ciencia de Datos:

- `/01_data_understanding`: Análisis Exploratorio de Datos (EDA) identificando desbalance de clases (8% impagos).
- `/02_data_preparation`: Ingeniería de características e integración de fuentes de datos (`application` + `bureau`).
- `/03_modeling`: Entrenamiento de modelo **XGBoost** optimizado con `scale_pos_weight` para manejo de desbalance.
- `/04_evaluation`: Generación de métricas de negocio (AUC-ROC, Matriz de Confusión) y gráficos de desempeño.
- `/05_deployment`: API REST productiva desarrollada con **FastAPI** para inferencia en tiempo real.
- `/artifacts`: Almacenamiento de modelos serializados (`.joblib`) y metadatos del pipeline.

##  Tecnologías Utilizadas
- **Lenguaje:** Python 3.11+
- **Modelado:** XGBoost, Scikit-Learn
- **Procesamiento de Datos:** Pandas, PyArrow
- **API & Despliegue:** FastAPI, Uvicorn, Pydantic
- **Visualización:** Matplotlib, Seaborn

##  Resultados del Modelo
El modelo final prioriza la detección de deudores para proteger el capital del banco:
- **AUC-ROC:** ~0.76 (Buena capacidad de discriminación).
- **Recall (Clase 1):** ~0.69 (El modelo detecta aprox. el 70% de los clientes riesgosos).
- **Estrategia:** Se penalizó fuertemente el error en la clase minoritaria.

##  Instrucciones de Instalación y Uso

1. **Instalar dependencias:**
   Ejecuta el siguiente comando en la terminal para instalar las librerías necesarias:
   ```bash
   pip install -r requirements.txt