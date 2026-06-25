# maternal-health-risk-predict

Trabajo práctico final para predecir el nivel de riesgo en salud materna a partir de signos vitales y variables clínicas básicas. El proyecto combina un modelo de machine learning entrenado con scikit-learn, una API con FastAPI y una interfaz web en Streamlit.

## Contexto

El objetivo del TP es clasificar el riesgo materno en tres niveles:

- Riesgo Bajo
- Riesgo Medio
- Riesgo Alto

La predicción se realiza con un árbol de decisión entrenado sobre el dataset `maternal-health-risk.csv`. El modelo toma como entrada estas variables:

- `Age`
- `SystolicBP`
- `DiastolicBP`
- `BS`
- `BodyTemp`
- `HeartRate`

La arquitectura del proyecto está separada en tres partes:

- notebooks para exploración y entrenamiento del modelo
- backend con FastAPI para servir la predicción
- frontend con Streamlit para interactuar con el modelo

## Introducción

Este TP busca mostrar un flujo completo de machine learning aplicado: preparación de datos, entrenamiento, exportación del modelo, exposición de una API y construcción de una interfaz visual para hacer consultas.

La salida del modelo no reemplaza un criterio médico. La idea es mostrar una prueba de concepto técnica orientada a clasificación y visualización de resultados.

## Estructura del proyecto

- `maternal-health-risk.csv`: dataset utilizado para entrenar y evaluar el modelo.
- `main.ipynb` y `main2.ipynb`: notebooks de exploración, entrenamiento y pruebas.
- `backend/`: API en FastAPI.
- `backend/main.py`: punto de entrada del backend.
- `backend/utils.py`: funciones de carga del modelo y predicción.
- `backend/models/`: artefactos generados por el entrenamiento.
- `frontend/`: aplicación en Streamlit.
- `frontend/app.py`: interfaz web principal.

## Inicialización del TP

Si vas a levantar el proyecto desde cero, seguí estos pasos:

### Requisitos previos
- Python 3.9 o superior
- pip
- Git (opcional, si clonas el repo)

### Pasos de instalación

#### 1. Cloná o abrí el repositorio en tu máquina
```bash
git clone <url-del-repo>
cd maternal-health-risk-predict
```

#### 2. Creá un entorno virtual de Python
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalá las dependencias

Para instalar el proyecto completo:
```bash
pip install -r requirements.txt
```

O instalá componentes por separado:

**Backend:**
```bash
pip install -r backend/requirements.txt
```

**Frontend:**
```bash
pip install -r frontend/requirements.txt
```

#### 4. Ejecutá el notebook de entrenamiento

Abre `main.ipynb` en Jupyter y ejecuta todas las celdas para:
- Explorar el dataset
- Preprocesar los datos
- Entrenar el árbol de decisión
- Exportar el modelo a `backend/models/decision_tree_maternal_risk.pkl`
- Generar métricas en `backend/models/metrics.json`

```bash
jupyter notebook main.ipynb
```

#### 5. Levantá el backend (en una terminal)

Desde la carpeta `backend/`:
```bash
cd backend
uvicorn main:app --reload
```

La API estará disponible en:
- Base: `http://127.0.0.1:8000`
- Documentación (Swagger): `http://127.0.0.1:8000/docs`

#### 6. Levantá el frontend (en otra terminal)

Desde la carpeta `frontend/`:
```bash
cd frontend
streamlit run app.py
```

La aplicación web abrirá automáticamente en `http://localhost:8501`

## Stack Tecnológico

### Backend
- **FastAPI**: Framework web moderno y de alto rendimiento
- **Uvicorn**: Servidor ASGI
- **scikit-learn**: Algoritmo de árbol de decisión
- **Joblib**: Serialización del modelo entrenado
- **Pydantic**: Validación de datos

### Frontend
- **Streamlit**: Framework para crear aplicaciones web de datos
- **Plotly**: Visualizaciones interactivas
- **Pandas**: Manipulación de datos
- **Requests**: Comunicación HTTP con la API

### Notebook
- **Pandas**: Exploración y manipulación de datos
- **Matplotlib** y **Seaborn**: Visualizaciones
- **scikit-learn**: Entrenamiento del modelo
- **Numpy**: Cálculos numéricos

## Endpoints de la API

### GET `/`
Estado general de la API.

### GET `/health`
Verifica que el servidor y el modelo están operativos.

### POST `/predict`
Realiza una predicción de riesgo.

**Parámetros (JSON):**
```json
{
  "Age": 28,
  "SystolicBP": 130,
  "DiastolicBP": 90,
  "BS": 7.5,
  "BodyTemp": 98.0,
  "HeartRate": 80
}
```

**Respuesta:**
```json
{
  "prediccion": 1,
  "riesgo": "Riesgo Medio",
  "probabilidades": {
    "Riesgo Bajo": 0.3333,
    "Riesgo Medio": 0.3333,
    "Riesgo Alto": 0.3334
  }
}
```

## Notas Importantes

- El modelo se exporta desde el notebook en formato `.pkl` (joblib).
- Asegurate de que el archivo `backend/models/decision_tree_maternal_risk.pkl` exista antes de levantar el backend.
- El frontend espera que el backend esté corriendo en `http://127.0.0.1:8000` por defecto. Podés cambiar esto con la variable de entorno `API_URL`.
- La API devuelve probabilidades para cada clase de riesgo, no solo la predicción más probable.


