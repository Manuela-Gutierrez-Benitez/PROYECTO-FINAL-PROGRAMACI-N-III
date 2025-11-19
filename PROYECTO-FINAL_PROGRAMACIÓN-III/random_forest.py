import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv("Crop_recommendationV2.csv")

X = df[['K', 'ph', 'humidity']]
y = df["label"]

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Columnas numéricas y categóricas
numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
categorical_cols = X.select_dtypes(include=["object","category"]).columns

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Pipeline RF
rf_model = Pipeline([
    ("preprocessing", preprocessor),
    ("rf", RandomForestClassifier(random_state=42))
])

# Entrenamiento inicial
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Accuracy inicial:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Grid Search
param_grid = {
    "rf__n_estimators": [200, 300, 400],
    "rf__max_depth": [10, 20, 30, None],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__bootstrap": [True, False]
}

grid_rf = GridSearchCV(
    rf_model, 
    param_grid, 
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("\nMejores hiperparámetros:", grid_rf.best_params_)
print("Mejor F1-macro:", grid_rf.best_score_)

# Predicción final
final_pred = grid_rf.predict(X_test)

print("\nAccuracy final:", accuracy_score(y_test, final_pred))
print(classification_report(y_test, final_pred))

joblib.dump(grid_rf.best_estimator_, "modelo_random_forest.pkl")

print("\nModelo guardado como 'modelo_random_forest.pkl'")
