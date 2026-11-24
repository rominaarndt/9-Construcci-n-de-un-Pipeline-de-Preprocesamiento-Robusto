import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Cargar dataset
df = pd.read_parquet("dataset_procesado.parquet")

# 2. Definir X e y
cols_a_excluir = [
    "is_fraud",
    "fraud_type",
    "transaction_id",
    "customer_id",
    "transaction_date",
]
cols_existentes = [c for c in cols_a_excluir if c in df.columns]

X = df.drop(columns=cols_existentes)
y = df["is_fraud"]

# 3. División Train / Test antes de transformar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Separar columnas numéricas y categóricas
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

# 5. Transformadores por tipo de variable
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 6. Pipeline final con manejo de desbalance de clases
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# 7. Entrenamiento, evaluación y persistencia
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(full_pipeline, "pipeline_completo.joblib")

# 8. Prueba de inferencia sobre datos nuevos
modelo_cargado = joblib.load("pipeline_completo.joblib")
predicciones_nuevas = modelo_cargado.predict(X_test.iloc[:5])
print("Predicciones sobre 5 registros nuevos:", predicciones_nuevas)
