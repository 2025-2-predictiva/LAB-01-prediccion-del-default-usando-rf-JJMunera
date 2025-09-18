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
df_test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")  
df_train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

df_test_data = df_test_data.rename(columns={"default payment next month": "default"})
df_train_data = df_train_data.rename(columns={"default payment next month": "default"})

df_test_data = df_test_data.drop(columns=["ID"]).dropna()
df_train_data = df_train_data.drop(columns=["ID"]).dropna()

df_test_data["EDUCATION"] = df_test_data["EDUCATION"].mask(df_test_data["EDUCATION"] > 4, 4)
df_train_data["EDUCATION"] = df_train_data["EDUCATION"].mask(df_train_data["EDUCATION"] > 4, 4)
df_test_data["EDUCATION"] = df_test_data["EDUCATION"].astype("category")
df_train_data["EDUCATION"] = df_train_data["EDUCATION"].astype("category")

#%%
# Paso 2.
x_train = df_train_data.drop(columns=["default"])
y_train = df_train_data["default"]
x_test = df_test_data.drop(columns=["default"])
y_test = df_test_data["default"]

#%%
# Paso 3.
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

pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
pipe.fit(x_train, y_train)

#%%
# Paso 4.
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import gzip, pickle, json, os

categorical_cols = x_train.select_dtypes(include=["object", "category"]).columns
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))

param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [None, 10],
}

grid = GridSearchCV(pipe, param_grid, cv=10, scoring="balanced_accuracy")
grid.fit(x_train, y_train)

#%%
# Paso 5.
os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)  # Guardar grid (no pipe)

#%%
# Paso 6.
y_train_pred = grid.predict(x_train)
y_test_pred = grid.predict(x_test)

metrics_train = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, y_train_pred),
    "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    "recall": recall_score(y_train, y_train_pred),
    "f1_score": f1_score(y_train, y_train_pred),
}

metrics_test = {
    "type": "metrics",
    "dataset": "test",
    "precision": precision_score(y_test, y_test_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "recall": recall_score(y_test, y_test_pred),
    "f1_score": f1_score(y_test, y_test_pred),
}

#%%
# Paso 7.
cm_train_arr = confusion_matrix(y_train, y_train_pred)
cm_test_arr = confusion_matrix(y_test, y_test_pred)

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

os.makedirs("files/output", exist_ok=True)
metrics = [metrics_train, metrics_test, cm_train, cm_test]
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for m in metrics:
        f.write(json.dumps(m) + "\n")

# %%
