# Pipeline de Preprocesamiento Robusto — Detección de Fraude

Este repositorio contiene la implementación de una arquitectura automatizada y reutilizable utilizando `Pipeline` y `ColumnTransformer` de **Scikit-Learn** para el preprocesamiento de datos y entrenamiento de un modelo de clasificación binario orientado a la detección de fraude en transacciones financieras.

---

## 📋 Descripción del Dataset

El proyecto utiliza el archivo `dataset_procesado.parquet`, generado previamente en la etapa de ingesta y limpieza estructural. Contiene registros de transacciones financieras con variables demográficas, comportamentales y transaccionales:

* **Variable objetivo (`target`):** `is_fraud` (`0` = Transacción legítima, `1` = Fraude). Presenta un marcado desbalance de clases (~94,5 % clase `0` frente a ~5,5 % clase `1`).
* **Variables numéricas:** `customer_age`, `credit_score`, `account_age_years`, `account_balance`, `transaction_amount`, `num_prev_transactions`, `transaction_freq_monthly`, `distance_from_home_km`, `time_since_last_txn_hrs`, `is_international`, `failed_attempts`, `pin_changed_recently`, entre otras métricas cuantitativas.
* **Variables categóricas:** `country`, `city`, `merchant_category`, `payment_method`, `device_type`, etc.
* **Columnas excluidas del entrenamiento:** `fraud_type` (para evitar fuga de información sobre el objetivo) e identificadores/fechas (`transaction_id`, `customer_id`, `transaction_date`).

---

## 🏗️ Arquitectura del Pipeline y Prevención de Data Leakage

Para garantizar que no exista fuga de datos (*data leakage*) y que las transformaciones sean idénticas en entrenamiento e inferencia, el flujo sigue esta estructura:

1. **División previa (`train_test_split`):** Se separan los conjuntos de entrenamiento (80 %) y prueba (20 %) con muestreo estratificado (`stratify=y`) **antes** de ajustar cualquier transformador.
2. **Preprocesamiento diferenciado (`ColumnTransformer`):**
   * **Pipeline Numérico:** Imputación de valores faltantes mediante la mediana (`SimpleImputer(strategy="median")`) seguida de estandarización (`StandardScaler()`).
   * **Pipeline Categórico:** Imputación por moda (`SimpleImputer(strategy="most_frequent")`) seguida de codificación One-Hot (`OneHotEncoder(handle_unknown="ignore")`) para soportar categorías nuevas en producción sin fallos.
3. **Estimador (`LogisticRegression`):** Se integra al final del `Pipeline` con `class_weight="balanced"` para compensar el desbalance de clases propio de los problemas de fraude.
4. **Persistencia:** El pipeline completo entrenado se exporta mediante `joblib` en el archivo `pipeline_completo.joblib`.

---

## 📊 Resultados de Evaluación

Al evaluar el pipeline sobre el conjunto de prueba (`200.000` registros: `188.949` legítimos y `11.051` fraudes) utilizando `class_weight="balanced"`, se obtuvieron las siguientes métricas:

| Clase | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (No Fraude)** | 0.97 | 0.71 | 0.82 | 188.949 |
| **1 (Fraude)** | 0.10 | 0.57 | 0.17 | 11.051 |
| **Accuracy Global** | | | **0.70** | **200.000** |

> **Nota sobre el desbalance:** Sin ponderación de clases, el modelo alcanzaba un 94 % de *accuracy* pero un *recall* de `0.00` en la clase `1`. Al incorporar `class_weight="balanced"`, el modelo logra detectar el **57 % de los fraudes reales**, priorizando la sensibilidad ante eventos fraudulentos.

---

## 🚀 Estructura del Proyecto y Ejecución

```text
pipeline-preprocesamiento/
├── dataset_procesado.parquet   # Dataset limpio de entrada
├── pipeline.py                 # Script principal con el Pipeline y evaluación
├── pipeline_completo.joblib    # Artefacto del pipeline entrenado y serializado
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación
```

### Cómo ejecutarlo localmente

1. Instalar las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar el script de entrenamiento, evaluación y prueba de inferencia:
   ```bash
   python pipeline.py
   ```
3. Para realizar predicciones sobre nuevos datos desde Python:
   ```python
   import joblib

   pipeline = joblib.load("pipeline_completo.joblib")
   predicciones = pipeline.predict(nuevos_datos)
   ```