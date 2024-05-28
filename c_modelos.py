import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from b_preprocesamiento import df_historicos,df_nuevos
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error

df_historicos.info()
df_nuevos.info()
df_historicos.isnull().sum()
df_nuevos.isnull().sum()

print(df_historicos['clusters'].value_counts())

### Preprocesamiento y alistamiento de los datos para el entretamiento del modelo clasificatorio

scaler = StandardScaler()

x_historicos = df_historicos.drop(['clusters', 'final_interest_rate', 'NoPaidPerc'], axis=1) 
y_historicos = df_historicos['clusters']
x_nuevos = df_nuevos.drop(['ID', 'NewLoanApplication'], axis=1)

# Identificar variables numéricas y categóricas
numeric_features = x_historicos.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x_historicos.select_dtypes(include=['object', 'category']).columns

# Preprocesamiento: escalar numéricas y crear variables dummy para categóricas
x_historicos[numeric_features] = StandardScaler().fit_transform(x_historicos[numeric_features])
x_historicos = pd.get_dummies(x_historicos, columns=categorical_features, drop_first=True)

# Identificar variables numéricas y categóricas
numeric_featuresN = x_nuevos.select_dtypes(include=['int64', 'float64']).columns
categorical_featuresN = x_nuevos.select_dtypes(include=['object', 'category']).columns

# Preprocesamiento: escalar numéricas y crear variables dummy para categóricas
x_nuevos[numeric_featuresN] = StandardScaler().fit_transform(x_nuevos[numeric_featuresN])
x_nuevos = pd.get_dummies(x_nuevos, columns=categorical_featuresN, drop_first=True)

# Evaluar varios modelos
# 20% para los datos de prueba y 80% para entrenamiento.

X_train, X_test, y_train, y_test = train_test_split(x_historicos, y_historicos, test_size=0.2, random_state=42)

# Definimos los modelos
models = {
    "logistic Regression": LogisticRegression(), # La regresión logística es un modelo simple y eficaz para problemas de clasificación binaria.
    "Gradient Boosting": GradientBoostingClassifier(), # Gradient Boosting es un método de ensamble que combina varios árboles de decisión débiles para crear un modelo robusto.
    "RandomForest Classifier": RandomForestClassifier(), # Modelos de regularización para manejar el problema de multicolinealidad y reducir el riesgo de sobreajuste en la regresión lineal
    "K-Nearest Neighbors": KNeighborsClassifier(), #KNN es un algoritmo simple y efectivo para problemas de clasificación.
    "DNaive Bayes": GaussianNB(), #El clasificador Naive Bayes es un modelo simple y rápido basado en el teorema de Bayes.
    "Gradient Boosting": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42), # MLP es una red neuronal artificial que puede modelar relaciones no lineales complejas.
}

# Evaluar los modelos
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Precision": result['weighted avg']['precision'],
        "Recall": result['weighted avg']['recall'],
        "F1-Score": result['weighted avg']['f1-score']
    })

# Convertir los resultados a un DataFrame para visualización
results_df = pd.DataFrame(results)

# Mostrar los resultados
print(results_df)

# Encontrar el modelo con la mejor métrica F1-Score
best_model = results_df.loc[results_df['F1-Score'].idxmax()]

# El mejor modelo segun las metricas analizadas es el logistic Regression

best_model = LogisticRegression()
best_model.fit(x_historicos, y_historicos)

y_pred = best_model.predict(x_nuevos)

#Asignamos los cluster en una nueva columna
df_nuevos['clusters'] = y_pred

### Ahora se buscara el mejor modelo para hacer la prediccion del riesgo

x2_historicos = df_historicos.drop(['clusters', 'final_interest_rate', 'NoPaidPerc'], axis=1) 
y2_historicos = df_historicos['NoPaidPerc']
x2_nuevos = df_nuevos.drop(['ID', 'NewLoanApplication', 'clusters'], axis=1)

# Identificar variables numéricas y categóricas
numeric_features2 = x2_historicos.select_dtypes(include=['int64', 'float64']).columns
categorical_features2 = x2_historicos.select_dtypes(include=['object', 'category']).columns

# Preprocesamiento: escalar numéricas y crear variables dummy para categóricas
x2_historicos[numeric_features2] = StandardScaler().fit_transform(x2_historicos[numeric_features2])
x2_historicos = pd.get_dummies(x2_historicos, columns=categorical_features2, drop_first=True)

# Identificar variables numéricas y categóricas
numeric_featuresN2 = x2_nuevos.select_dtypes(include=['int64', 'float64']).columns
categorical_featuresN2 = x2_nuevos.select_dtypes(include=['object', 'category']).columns

# Preprocesamiento: escalar numéricas y crear variables dummy para categóricas
x2_nuevos[numeric_featuresN2] = StandardScaler().fit_transform(x2_nuevos[numeric_featuresN2])
x2_nuevos = pd.get_dummies(x2_nuevos, columns=categorical_featuresN2, drop_first=True)

# Evaluar varios modelos
# 20% para los datos de prueba y 80% para entrenamiento.
X_train, X_test, y_train, y_test = train_test_split(x2_historicos, y2_historicos, test_size=0.2, random_state=42)

# Definimos los modelos
models = {
    "Linear Regression": LinearRegression(), 
    "Random Forest": RandomForestRegressor(), 
    "Decision Tree": DecisionTreeRegressor(), 
    "Gradient Boosting": GradientBoostingRegressor(), 
    "Support Vector Machines": SVR(), 
    "K-Nearest Neighbors": KNeighborsRegressor() 
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MSE": mse, 
        "RMSE": rmse,
        "R2": r2, 
        "MAE": mae, 
    })

# Convertir los resultados a un DataFrame para visualización
results_df = pd.DataFrame(results)

# Mostrar los resultados
print(results_df)

# Encontrar el modelo con la mejor métrica R2
best_model = results_df.loc[results_df['R2'].idxmax()]

# El mejor modelo segun las metricas analizadas es el Gradient Boosting
best_model = GradientBoostingRegressor()
best_model.fit(x2_historicos, y2_historicos)

y_pred = best_model.predict(x2_nuevos)

#Asignamos los riesgos en una nueva columna
df_nuevos['riesgos_pred'] = y_pred

# Para cada clúster, calcular el riesgo promedio de incumplimiento
cluster_risk = df_nuevos.groupby('clusters')['riesgos_pred'].mean()

# Definir los componentes del interés
operational_cost_interest = 0.05
expected_margin_interest = 0.10

# Calcular la tasa de interés para cada clúster
cluster_interest_rate = np.round(operational_cost_interest + expected_margin_interest + cluster_risk, 2)

# Asignar la tasa de interés calculada a cada cliente en función de su clúster
df_nuevos['final_interest_rate'] = df_nuevos['clusters'].apply(lambda x: cluster_interest_rate[x])