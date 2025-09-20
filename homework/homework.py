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
df_test_data = pd.read_csv("../files/input/test_data.csv.zip", compression="zip")  
df_train_data = pd.read_csv("../files/input/train_data.csv.zip", compression="zip")

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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier

#Columnas categóricas
categoric_vars = ["SEX", "MARRIAGE", "EDUCATION"]
numeric_vars = [n for n in x_train.columns if n not in categoric_vars]

preprocessor= ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categoric_vars),
        ('num', StandardScaler(with_mean=True, with_std=True), numeric_vars)
    ]
)
pipeline= Pipeline(
    [
        ('preprocessor', preprocessor),
        ('pca', PCA()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=42))
    ]
)

#%%
# Paso 4.
import pickle
import os
import gzip
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

param_grid = {
    "pca__n_components": [20, x_train.shape[1]-2],
    "feature_selection__k": [20],
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}

model=GridSearchCV(
    pipeline, 
    param_grid, 
    cv=10, 
    scoring="balanced_accuracy",
    n_jobs = -1,
    refit = True,
    )

model.fit(x_train, y_train)

models_dir='../files/models/'
os.makedirs(models_dir, exist_ok=True)

#nombre del archivo donde se guardará el modelo
compressed_model_path = "../files/models/homework_model.pkl.gz"

with gzip.open(compressed_model_path, 'wb') as f:
    pickle.dump(model, f)

#%%
# Paso 6.
import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas para el conjunto de entrenamiento
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }

    # Calcular métricas para el conjunto de prueba
    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }

    # Crear carpeta si no existe
    output_dir = '../files/output'
    os.makedirs(output_dir, exist_ok=True)

    # Guardar las métricas en un archivo JSON
    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:  # Usar 'w' para comenzar con un archivo limpio
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')
#%%
# Paso 7.
from sklearn.metrics import confusion_matrix


# Función para calcular las matrices de confusión y guardarlas
def calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Convertir las matrices de confusión en formato JSON
    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]

    # Guardar las matrices de confusión en el mismo archivo JSON
    output_path = '../files/output/metrics.json'
    with open(output_path, 'a') as f:  # Usar 'a' para agregar después de las métricas
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

# Función principal para ejecutar todo
def main(model, X_train, X_test, y_train, y_test):
    # Crear el directorio de salida si no existe
    import os
    os.makedirs('../files/output', exist_ok=True)

    # Calcular y guardar las métricas
    calculate_and_save_metrics(model, X_train, X_test, y_train, y_test)

    # Calcular y guardar las matrices de confusión
    calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test)
#%%