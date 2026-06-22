"""
utils.py
Funciones auxiliares para cargar el modelo y transformar sus predicciones
en algo legible para el frontend.
"""

import os
import joblib

# Ruta al modelo exportado desde el notebook (notebook -> joblib -> acá)
RUTA_MODELO = os.path.join(os.path.dirname(__file__), "models", "decision_tree_maternal_risk.pkl")

# Mismo orden de columnas que se usó para entrenar el modelo en el notebook.
# Si esto no coincide en orden y cantidad, el modelo va a predecir cualquier cosa.
COLUMNAS_MODELO = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# Traducimos la salida numérica del modelo (0, 1, 2) a algo entendible
ETIQUETAS_RIESGO = {
    0: "Riesgo Bajo",
    1: "Riesgo Medio",
    2: "Riesgo Alto",
}


def cargar_modelo():
    """Carga el modelo .pkl entrenado con joblib. Se llama una sola vez al iniciar el server."""
    if not os.path.exists(RUTA_MODELO):
        raise FileNotFoundError(
            f"No se encontró el modelo en {RUTA_MODELO}. "
            "Asegurate de haber corrido el notebook y exportado el .pkl en la carpeta models/."
        )
    return joblib.load(RUTA_MODELO)


def predecir_riesgo(modelo, datos: dict) -> dict:
    """
    Recibe un diccionario con las 6 variables del paciente, arma la fila en el
    orden correcto y devuelve la predicción junto con las probabilidades de cada clase.
    """
    fila = [[datos[col] for col in COLUMNAS_MODELO]]

    prediccion = int(modelo.predict(fila)[0])
    probabilidades = modelo.predict_proba(fila)[0]

    return {
        "prediccion": prediccion,
        "riesgo": ETIQUETAS_RIESGO[prediccion],
        "probabilidades": {
            ETIQUETAS_RIESGO[clase]: round(float(prob), 4)
            for clase, prob in zip(modelo.classes_, probabilidades)
        },
    }
