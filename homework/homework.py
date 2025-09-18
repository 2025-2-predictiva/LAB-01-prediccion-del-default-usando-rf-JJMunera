# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#%%
import pandas as pd
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
df_test_data = pd.read_csv("../files/input/test_data.csv.zip", compression="zip")  
df_train_data = pd.read_csv("../files/input/train_data.csv.zip", compression="zip")

df_test_data = df_test_data.rename(columns={"default payment next month": "default"})
df_train_data = df_train_data.rename(columns={"default payment next month": "default"})
# - Remueva la columna "ID".
df_test_data = df_test_data.drop(columns=["ID"])
df_train_data = df_train_data.drop(columns=["ID"])
# - Elimine los registros con informacion no disponible.
df_test_data = df_test_data.dropna()
df_train_data = df_train_data.dropna()
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
df_test_data["EDUCATION"] = df_test_data["EDUCATION"].mask(df_test_data["EDUCATION"] > 4, 4)
df_train_data["EDUCATION"] = df_train_data["EDUCATION"].mask(df_train_data["EDUCATION"] > 4, 4)
df_test_data["EDUCATION"] = df_test_data["EDUCATION"].astype("category")
df_train_data["EDUCATION"] = df_train_data["EDUCATION"].astype("category")

print(df_test_data.head())

#%%
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = df_train_data.drop(columns=["default"])
y_train = df_train_data["default"]
x_test = df_test_data.drop(columns=["default"])
y_test = df_test_data["default"]
# %%
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

categoric_vars = ["SEX", "MARRIAGE", "EDUCATION"]
numeric_vars = [n for n in x_train.columns if n not in categoric_vars]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categoric_vars),
        ('num', 'passthrough', numeric_vars)
    ]
)

pipe = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=0)
)
pipe.fit(x_train, y_train)
# %%
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import gzip
import pickle
import json
import numpy as np

# Suponiendo que ya tienes x_train, y_train, x_test, y_test definidos
# Reemplaza estas variables si provienen de otra parte:
# x_train, y_train, x_test, y_test = ...

# Detectar columnas categóricas automáticamente o defínelas:
categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Hiperparámetros para GridSearchCV
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10],
}

grid = GridSearchCV(pipe, param_grid, cv=10, scoring="balanced_accuracy")
grid.fit(x_train, y_train)

# %%
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

# %%
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall y f1-score
# para los conjuntos de entrenamiento y prueba y guárdelas en files/output/metrics.json.
y_train_pred = grid.predict(x_train)
y_test_pred = grid.predict(x_test)

metrics_train = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, y_train_pred, average="weighted"),
    "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred, average="weighted"),
    "f1_score": f1_score(y_train, y_train_pred, average="weighted"),
}

metrics_test = {
    "type": "metrics",
    "dataset": "test",
    "precision": precision_score(y_test, y_test_pred, average="weighted"),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "recall": recall_score(y_test, y_test_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_test_pred, average="weighted"),
}

# %%
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y prueba
# en el formato requerido.
cm_train_arr = confusion_matrix(y_train, y_train_pred)
cm_test_arr = confusion_matrix(y_test, y_test_pred)

# Suponemos un problema binario con etiquetas [0,1]
cm_train = {
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {"predicted_0": int(cm_train_arr[0, 0]), "predicted_1": int(cm_train_arr[0, 1])},
    "true_1": {"predicted_0": int(cm_train_arr[1, 0]), "predicted_1": int(cm_train_arr[1, 1])},
}

cm_test = {
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {"predicted_0": int(cm_test_arr[0, 0]), "predicted_1": int(cm_test_arr[0, 1])},
    "true_1": {"predicted_0": int(cm_test_arr[1, 0]), "predicted_1": int(cm_test_arr[1, 1])},
}

# Guardar todo en JSON Lines en el orden correcto
metrics = [metrics_train, metrics_test, cm_train, cm_test]

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for m in metrics:
        f.write(json.dumps(m) + "\n")
#%%