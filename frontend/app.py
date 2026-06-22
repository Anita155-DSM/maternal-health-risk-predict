"""
app.py
Frontend en Streamlit que consume la API de FastAPI (ver carpeta backend/)
para predecir el riesgo en salud materna y mostrar cómo rinde el modelo.

Para correrlo (con el backend ya levantado en otra terminal):
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Riesgo en Salud Materna",
    page_icon="./dataset-img.png",
    layout="wide",
)

st.title("Predicción de Riesgo en Salud Materna")
st.caption(
    "Trabajo Práctico Final - Taller de Lenguajes de Programación III. "
    "Modelo: Árbol de Decisión (scikit-learn) servido vía FastAPI."
)

tab_prediccion, tab_metricas = st.tabs(["🔮 Predicción", "📊 Rendimiento del modelo"])


# ---------------------------------------------------------------------------
# TAB 1: Predicción
# ---------------------------------------------------------------------------
with tab_prediccion:
    st.subheader("Ingresá los signos vitales del paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Edad (años)", min_value=10, max_value=70, value=28, step=1)
        systolic_bp = st.number_input("Presión sistólica (mmHg)", min_value=60, max_value=200, value=120, step=1)

    with col2:
        diastolic_bp = st.number_input("Presión diastólica (mmHg)", min_value=40, max_value=140, value=80, step=1)
        bs = st.number_input("Glucosa en sangre (mmol/L)", min_value=4.0, max_value=25.0, value=7.5, step=0.1)

    with col3:
        body_temp = st.number_input("Temperatura corporal (°F)", min_value=95.0, max_value=105.0, value=98.0, step=0.1)
        heart_rate = st.number_input("Frecuencia cardíaca (lpm)", min_value=30, max_value=180, value=75, step=1)

    st.write("")

    if st.button("Predecir nivel de riesgo", type="primary", use_container_width=True):
        payload = {
            "Age": age,
            "SystolicBP": systolic_bp,
            "DiastolicBP": diastolic_bp,
            "BS": bs,
            "BodyTemp": body_temp,
            "HeartRate": heart_rate,
        }

        try:
            respuesta = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            respuesta.raise_for_status()
            resultado = respuesta.json()

            riesgo = resultado["riesgo"]
            probabilidades = resultado["probabilidades"]

            colores = {
                "Riesgo Bajo": "#2ECC71",
                "Riesgo Medio": "#F39C12",
                "Riesgo Alto": "#E74C3C",
            }

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                st.markdown(
                    f"""
                    <div style="background-color:{colores[riesgo]}22; border-left: 6px solid {colores[riesgo]};
                                padding: 1.2rem; border-radius: 0.5rem;">
                        <p style="font-size:1rem; margin-bottom:0.2rem; color:gray;">Predicción del modelo</p>
                        <p style="font-size:2rem; font-weight:bold; margin:0; color:{colores[riesgo]};">{riesgo}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                df_prob = pd.DataFrame({
                    "Nivel de riesgo": list(probabilidades.keys()),
                    "Probabilidad": list(probabilidades.values()),
                })
                fig = px.bar(
                    df_prob, x="Probabilidad", y="Nivel de riesgo", orientation="h",
                    color="Nivel de riesgo",
                    color_discrete_map=colores,
                    range_x=[0, 1],
                    text_auto=".0%",
                )
                fig.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error(
                f"No se pudo conectar con la API en {API_URL}. "
                "¿La levantaste con `uvicorn main:app --reload` en la carpeta backend/?"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"La API devolvió un error: {e.response.json()}")


# ---------------------------------------------------------------------------
# TAB 2: Métricas / rendimiento del modelo
# ---------------------------------------------------------------------------
with tab_metricas:
    st.subheader("¿Qué tan bien predice el modelo?")
    st.caption("Métricas calculadas sobre el conjunto de test (datos que el modelo nunca vio durante el entrenamiento).")

    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=10)
        resp.raise_for_status()
        m = resp.json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Profundidad del árbol", m["max_depth"])
        c3.metric("Muestras de entrenamiento", m["n_train"])
        c4.metric("Muestras de test", m["n_test"])

        st.markdown("---")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Matriz de confusión")
            cm = m["confusion_matrix"]
            labels = m["labels"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale="Blues", text=cm, texttemplate="%{text}",
                showscale=False,
            ))
            fig_cm.update_layout(
                xaxis_title="Predicción del modelo",
                yaxis_title="Valor real",
                yaxis_autorange="reversed",
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_der:
            st.markdown("#### Importancia de cada variable")
            imp = m["feature_importances"]
            df_imp = pd.DataFrame({
                "Variable": list(imp.keys()),
                "Importancia": list(imp.values()),
            }).sort_values("Importancia", ascending=True)
            fig_imp = px.bar(df_imp, x="Importancia", y="Variable", orientation="h")
            fig_imp.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Detalle por clase (precisión, recall, f1-score)")
        rep = m["classification_report"]
        filas = []
        for clase in m["labels"]:
            filas.append({
                "Clase": clase,
                "Precisión": round(rep[clase]["precision"], 2),
                "Recall": round(rep[clase]["recall"], 2),
                "F1-score": round(rep[clase]["f1-score"], 2),
                "Soporte (casos en test)": int(rep[clase]["support"]),
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        st.info(
            "💡 **Nota:** la clase 'Riesgo Medio' tiene el recall más bajo, lo que indica que el modelo "
            "tiene más dificultad para distinguirla de las otras dos. Es un punto de mejora a futuro "
            "(por ejemplo, probando Random Forest o balanceando las clases)."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            f"No se pudo conectar con la API en {API_URL}. "
            "¿La levantaste con `uvicorn main:app --reload` en la carpeta backend/?"
        )
    except requests.exceptions.HTTPError:
        st.warning(
            "El backend no tiene métricas precalculadas todavía (falta `backend/models/metrics.json`)."
        )
