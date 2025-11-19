import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("C:/Users/F3rna/OneDrive/Escritorio/Crop_recommendationV2.csv")

y = df["label"]

exclude = ["label", "soil_type", "growth_stage", "water_source_type"]
X = df.drop(columns=exclude)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Selección de columnas
numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Modelo SVM en un Pipeline
svm_model = Pipeline([
    ("preprocessing", preprocessor),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42))
])

# Entrenar
svm_model.fit(X_train, y_train)

# Predecir
y_pred = svm_model.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print("Accuracy inicial:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Grid Search para SVM
param_grid_svm = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01, 0.001],
    "svm__kernel": ["rbf"]
}

grid_svm = GridSearchCV(
    svm_model,
    param_grid_svm,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_svm.fit(X_train, y_train)

print("Mejores hiperparámetros:", grid_svm.best_params_)
print("Mejor score:", grid_svm.best_score_)
