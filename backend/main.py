"""
main.py
API construida con FastAPI que sirve el modelo de árbol de decisión
entrenado en el notebook (models/decision_tree_maternal_risk.pkl)
para predecir el nivel de riesgo en salud materna.

Para correrlo localmente:
    uvicorn main:app --reload

Documentación interactiva (Swagger) una vez levantado:
    http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import cargar_modelo, predecir_riesgo

# Acá vamos a guardar el modelo una vez cargado, para no leerlo del disco en cada request
modelo_ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta una sola vez, al levantar el servidor
    modelo_ml["model"] = cargar_modelo()
    print("Modelo cargado correctamente.")
    yield
    # Acá podríamos liberar recursos al apagar el server, no hace falta en este caso
    modelo_ml.clear()


app = FastAPI(
    title="API - Predicción de Riesgo en Salud Materna",
    description="Sirve un modelo de árbol de decisión (scikit-learn) entrenado para "
                 "clasificar el nivel de riesgo de una persona embarazada en base a sus signos vitales.",
    version="1.0.0",
    lifespan=lifespan,
)

# Habilitamos CORS para que el frontend (Streamlit, React, lo que sea) pueda
# llamar a esta API sin que el navegador lo bloquee. En producción conviene
# restringir allow_origins a la URL real del frontend en vez de "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DatosPaciente(BaseModel):
    """Datos de entrada que espera el modelo, con rangos físicamente razonables."""
    Age: int = Field(..., ge=10, le=70, description="Edad en años")
    SystolicBP: int = Field(..., ge=60, le=200, description="Presión sistólica (mmHg)")
    DiastolicBP: int = Field(..., ge=40, le=140, description="Presión diastólica (mmHg)")
    BS: float = Field(..., ge=4.0, le=25.0, description="Glucosa en sangre (mmol/L)")
    BodyTemp: float = Field(..., ge=95.0, le=105.0, description="Temperatura corporal (°F)")
    HeartRate: int = Field(..., ge=30, le=180, description="Frecuencia cardíaca (lpm)")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 28,
                "SystolicBP": 130,
                "DiastolicBP": 90,
                "BS": 7.5,
                "BodyTemp": 98.0,
                "HeartRate": 80,
            }
        }


class RespuestaPrediccion(BaseModel):
    prediccion: int
    riesgo: str
    probabilidades: dict


@app.get("/")
def home():
    return {"status": "ok", "mensaje": "API de riesgo en salud materna funcionando. Ver /docs"}


@app.get("/health")
def health_check():
    """Endpoint simple para chequear que el server y el modelo están operativos."""
    return {"status": "ok", "modelo_cargado": "model" in modelo_ml}


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(datos: DatosPaciente):
    """Recibe los signos vitales de una persona embarazada y devuelve el nivel de riesgo predicho."""
    if "model" not in modelo_ml:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")

    resultado = predecir_riesgo(modelo_ml["model"], datos.model_dump())
    return resultado
