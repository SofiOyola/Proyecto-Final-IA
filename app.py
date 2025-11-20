import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gdown
import os

# MODELOS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


# ===============================================
# CONFIGURACIÓN INICIAL
# ===============================================
st.set_page_config(page_title="Clasificación de Fallos", layout="wide")

st.title("Sistema Inteligente para Clasificación de Fallos en Transporte Público")

st.write("""
Aplicación interactiva para **EDA, preprocesamiento, entrenamiento de modelos ML y predicción en tiempo real**, basada fielmente en el proceso implementado en Google Colab.
""")


# -----------------------------------------------
# CREAR CARPETA DE MODELOS
# -----------------------------------------------
os.makedirs("saved_models", exist_ok=True)


# ===============================================
# 1. CARGA DEL DATASET
# ===============================================
st.header("1. Cargar Dataset")

file_id = st.text_input("Ingresa el ID del archivo en Google Drive (opcional)")

@st.cache_data
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "dataset.csv"
    gdown.download(url, output, quiet=True)
    df = pd.read_csv(output)
    return df

uploaded_file = st.file_uploader("O subir archivo CSV manualmente", type=["csv"])

df = None
if file_id:
    df = load_csv_from_drive(file_id)
    st.success("Dataset descargado desde Drive ✔")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset cargado exitosamente ✔")

if df is None:
    st.warning("Por favor carga un dataset para continuar.")
    st.stop()

# Vista previa
st.subheader("Vista previa del dataset")
st.write(df.head())
st.write("Dimensiones:", df.shape)
st.dataframe(df.dtypes)


# ===============================================
# 2. PREPROCESAMIENTO
# ===============================================
st.header("2. Preprocesamiento del Dataset")

if "falla" not in df.columns:
    st.error("La columna 'falla' NO existe. Debes usar *dataset_final_supervisado.csv* del Colab.")
    st.stop()

# Eliminar columnas del Colab
columns_to_drop = ["timestamp", "Unnamed: 0"]
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

st.success("Preprocesamiento aplicado ✔")

# Separar X y y
X = df.drop(columns=["falla"])
y = df["falla"]

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado para modelos que lo requieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================================
# 3. SELECCIÓN DEL MODELO
# ===============================================
st.header("3. Selección del Modelo")

model_name = st.selectbox(
    "Selecciona un modelo:",
    ["KNN", "Logistic Regression", "Random Forest", "SVM (LinearSVC)", "XGBoost"]
)

# Crear modelo según hiperparámetros optimizados del Colab
def get_model():
    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=3, metric="euclidean"
        )

    elif model_name == "Logistic Regression":
        return LogisticRegression(
            C=0.000123631882770522,
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced"
        )

    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=70,
            max_depth=5,
            min_samples_split=6,
            min_samples_leaf=4,
            criterion="gini",
            random_state=42
        )

    elif model_name == "SVM (LinearSVC)":
        return LinearSVC(
            C=2.550264850403285e-05,
            loss="squared_hinge",
            penalty="l2",
            max_iter=20000
        )

    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=290,
            max_depth=8,
            learning_rate=0.1740130838029839,
            subsample=0.9369139098379994,
            colsample_bytree=0.7246844304357644,
            gamma=0.10401360423556216,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )

model = get_model()


# ===============================================
# VARIABLES DE SESIÓN
# ===============================================
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None

if "trained_scaler" not in st.session_state:
    st.session_state["trained_scaler"] = None


# ===============================================
# 4. ENTRENAMIENTO
# ===============================================
st.header("4. Entrenamiento del Modelo")

if st.button("Entrenar Modelo"):
    
    # Selección de datos escalados
    if model_name in ["KNN", "Logistic Regression", "SVM (LinearSVC)"]:
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
        st.session_state["trained_scaler"] = scaler
    else:
        X_train_used = X_train
        X_test_used = X_test
        st.session_state["trained_scaler"] = None

    # Entrenar
    model.fit(X_train_used, y_train)

    # Guardar en sesión
    st.session_state["trained_model"] = model

    st.success("Modelo entrenado y almacenado en memoria ✔")

    # Evaluación
    y_pred = model.predict(X_test_used)
    acc = accuracy_score(y_test, y_pred)

    st.write("### Accuracy del modelo:", acc)
    st.text(classification_report(y_test, y_pred))

    # Matriz de Confusión
    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)


# ===============================================
# 5. PREDICCIÓN EN TIEMPO REAL
# ===============================================
st.header("5. Predicción en Tiempo Real")

st.write("Introduce los valores de las variables:")

input_values = []
for col in X.columns:
    value = st.number_input(col, value=0.0)
    input_values.append(value)

if st.button("Predecir"):

    if st.session_state["trained_model"] is None:
        st.error("⚠ Debes entrenar el modelo antes de predecir.")
        st.stop()

    model = st.session_state["trained_model"]
    scaler_session = st.session_state["trained_scaler"]

    arr = np.array(input_values).reshape(1, -1)

    # Escalar si corresponde
    if scaler_session is not None:
        arr = scaler_session.transform(arr)

    pred = model.predict(arr)[0]

    # Interpretación de la predicción
    if pred == 1:
        st.error("⚠ **El sistema presenta una FALLA.**")
    else:
        st.success("✔ **El sistema funciona correctamente (SIN FALLA).**")

