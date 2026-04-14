import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Crear datos de ejemplo (Simulando un dataset de MLOps)
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 2. Dividir datos
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Entrenar el modelo
print("Entrenando el modelo...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Guardar el modelo (Serialización)
joblib.dump(model, 'model.joblib')
print("¡Modelo guardado como model.joblib!")
