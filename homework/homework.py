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
from sklearn.model_selection import cross_validate
result = cross_validate(pipe, x_train, y_train, cv=10, scoring="balanced_accuracy")
print("Balanced accuracy (CV=10):", result['test_score'].mean())
#%%
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
import gzip
import pickle
import os
os.makedirs("files/models", exist_ok=True)
model_path = "files/models/model.pkl.gz"
with gzip.open(model_path, 'wb') as f:
    pickle.dump(pipe, f)
#%%
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

import json
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
# Asegura la carpeta de salida
os.makedirs("files/output", exist_ok=True)

y_train_pred = pipe.predict(x_train)
y_test_pred = pipe.predict(x_test)

def calcular_metricas(y_true, y_pred, dataset_name):
    return {
        'dataset': dataset_name,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

metrics_train = calcular_metricas(y_train, y_train_pred, 'train')
metrics_test = calcular_metricas(y_test, y_test_pred, 'test')
#%%
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def matriz_confusion(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {
            'predicted_0': int(cm[0,0]),
            'predicted_1': int(cm[0,1])
        },
        'true_1': {
            'predicted_0': int(cm[1,0]),
            'predicted_1': int(cm[1,1])
        }
    }

cm_train = matriz_confusion(y_train, y_train_pred, 'train')
cm_test = matriz_confusion(y_test, y_test_pred, 'test')

# --- 4️⃣ Guardar todo en metrics.json ---
metrics = [metrics_train, metrics_test, cm_train, cm_test]

with open("files/output/metrics.json", "w") as f:
    for m in metrics:
        f.write(json.dumps(m) + "\n")
#%%