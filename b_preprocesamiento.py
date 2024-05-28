import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # O RandomForestRegressor para un problema de regresión
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

historicos='https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
nuevos= 'https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'

df_historicos=pd.read_csv(historicos)
df_nuevos=pd.read_csv(nuevos)

df_historicos = df_historicos.drop('ID', axis=1)  

####### Evaluacion de variables del registro historico y su relación con respecto a la variable NoPaidPerc

# Seleccionar solo las variables numéricas
numerical_df = df_historicos.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación de todas las variables numéricas
correlation_matrix = numerical_df.corr()

# Crear un gráfico de calor (heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)

#### Analisis bivariado de todas las variables con respecto a la objetivo

porcentajenopago = 'NoPaidPerc'

# Seleccionar las columnas numéricas
numerical_cols = df_historicos.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Seleccionar las columnas categóricas
categorical_cols = df_historicos.select_dtypes(include=['object']).columns.tolist()

# Remover la columna objetivo de las columnas numéricas
numerical_cols.remove(porcentajenopago)

# Gráficos de torta para todas las variables categóricas
for col in categorical_cols:
    plt.figure(figsize=(8, 8))
    df_historicos[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title(f'Grafico de torta para {col}')
    plt.ylabel('')  # Ocultar el label del eje Y
    plt.show()

###### Boxplot
# Gráficos de caja para todas las variables categóricas con la variable objetivo
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_historicos[col], y=df_historicos[porcentajenopago])
    plt.title(f'Grafico boxplt de {porcentajenopago} vs {col}')
    plt.xticks(rotation=45)
    plt.show()

# Gráficos de violín para todas las variables categóricas con la variable objetivo
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df_historicos[col], y=df_historicos[porcentajenopago])
    plt.title(f'Grafico de violin {porcentajenopago} vs {col}')
    plt.xticks(rotation=45)
    plt.show()

# Gráficos de dispersión para todas las variables numéricas con la variable objetivo
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_historicos[col], y=df_historicos[porcentajenopago])
    plt.title(f'grafico de dispercion de {porcentajenopago} vs {col}')
    plt.show()

# Graficos de regresion

for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_historicos[col], y=df_historicos[porcentajenopago], line_kws={"color": "red"})
    plt.title(f'Grafico de regresion {porcentajenopago} vs {col}')
    plt.show()

# Separar características (X) y objetivo (y)
X = df_historicos.drop(['NoPaidPerc'], axis=1)  
y = df_historicos['NoPaidPerc']

# Identificar variables numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Preprocesamiento: escalar numéricas y crear variables dummy para categóricas
X[numeric_features] = StandardScaler().fit_transform(X[numeric_features])
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo de Bosque Aleatorio
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Usa RandomForestRegressor para regresión
model.fit(X_train, y_train)

# Selección de características usando SelectFromModel
selector = SelectFromModel(model, threshold='median')
selector.fit(X_train, y_train)

# Obtener las características seleccionadas
selected_features = X_train.columns[selector.get_support()]

# Crear un nuevo DataFrame con las características seleccionadas
X_selected = X[selected_features]

# Visualización de la importancia de las características seleccionadas
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importancia de las Características')
plt.show()

# Seleccionar las 10 variables más importantes
top_10_features = importance_df.head(10)['Feature'].tolist()
X_selected = X[top_10_features]

# Visualización de la importancia de las características
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Características por Importancia')
plt.show()

# Agregar varible NoPaidPerc al dataset de las variables seleccionadas para el clutering

X_selected['NoPaidPerc'] = df_historicos['NoPaidPerc']

# Determinación del número de clusters
inertia = []
silhouette_scores = []

range_clusters = range(2, 11)
for n_clusters in range_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(X_selected)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_selected, cluster_labels))

# Método del Codo
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# Calcula la segunda derivada de la inercia
acceleration = np.diff(inertia, 2)

# Encuentra el punto de codo (máximo cambio de aceleración)
optimal_k = np.argmax(acceleration) + 2  # El +2 es porque hemos hecho dos diferencias

# Graficar la curva de codo y el punto óptimo
plt.figure(figsize=(10, 6))
plt.plot(range_clusters, inertia, 'bo-')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Curva de codo para la determinación del número óptimo de clusters')
plt.axvline(x=optimal_k, color='r', linestyle='--', label='Número óptimo de clusters: {}'.format(optimal_k))
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=optimal_k)
clusters = kmeans.fit_predict(X_selected)

df_historicos['clusters'] = clusters

plt.hist(df_historicos['NoPaidPerc'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de la variable')
plt.show()

print(df_historicos['clusters'].value_counts())

# Para cada clúster, calcular el riesgo promedio de incumplimiento
cluster_risk = df_historicos.groupby('clusters')['NoPaidPerc'].mean()
cluster_age = df_historicos.groupby('clusters')['Age'].mean()
cluster_Assets = df_historicos.groupby('clusters')['Assets'].mean()
cluster_CreditScore = df_historicos.groupby('clusters')['CreditScore'].mean()

# Definir los componentes del interés
operational_cost_interest = 0.05
expected_margin_interest = 0.10

# Calcular la tasa de interés para cada clúster
cluster_interest_rate = operational_cost_interest + expected_margin_interest + cluster_risk

# Asignar la tasa de interés calculada a cada cliente en función de su clúster
df_historicos['final_interest_rate'] = df_historicos['clusters'].apply(lambda x: cluster_interest_rate[x])