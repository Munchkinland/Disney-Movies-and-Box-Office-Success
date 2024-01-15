import pandas as pd


# Cargar los conjuntos de datos
voice_actors_df = pd.read_csv("../data/raw/disney-voice-actors.csv")
director_df = pd.read_csv("../data/raw/disney-director.csv")
revenue_stream_2016_2019_df = pd.read_csv("../data/raw/disney_revenue_stream2016_2019.csv")
revenue_1991_2016_df = pd.read_csv("../data/raw/disney_revenue_1991-2016.csv")
total_gross_df = pd.read_csv("../data/raw/disney_movies_total_gross.csv")

# Ver las primeras filas de cada conjunto de datos para comprender su estructura
dfs = {
    "Voice Actors": voice_actors_df,
    "Directors": director_df,
    "Revenue Stream 2016-2019": revenue_stream_2016_2019_df,
    "Revenue 1991-2016": revenue_1991_2016_df,
    "Total Gross": total_gross_df
}

# Crear una función para mostrar un resumen de cada conjunto de datos
def get_df_summary(df_dict):
    summaries = {}
    for name, df in df_dict.items():
        summaries[name] = {
            "Primeras filas": df.head(),
            "Descripción": df.describe(include='all'),
            "Información": df.info()
        }
    return summaries

df_summaries = get_df_summary(dfs)
df_summaries

# Verificar la consistencia de los nombres de las películas entre los diferentes conjuntos de datos
# Comparamos los títulos de las películas en 'Total Gross' con los de 'Voice Actors' y 'Directors'

# Normalizamos los títulos de las películas para la comparación
total_gross_movies = total_gross_df['movie_title'].str.lower().unique()
voice_actors_movies = voice_actors_df['movie'].str.lower().unique()
directors_movies = director_df['name'].str.lower().unique()

# Verificar la intersección de películas entre los conjuntos de datos
common_movies_voice_actors = set(total_gross_movies).intersection(set(voice_actors_movies))
common_movies_directors = set(total_gross_movies).intersection(set(directors_movies))

# Contar el número de películas en común
num_common_movies_voice_actors = len(common_movies_voice_actors)
num_common_movies_directors = len(common_movies_directors)

num_common_movies_voice_actors, num_common_movies_directors

# Fusionar los conjuntos de datos
# Fusionar 'Voice Actors' y 'Total Gross'
merged_voice_actors = pd.merge(total_gross_df, voice_actors_df, 
                               left_on='movie_title', right_on='movie', how='inner')

# Fusionar 'Directors' y 'Total Gross'
merged_directors = pd.merge(total_gross_df, director_df, 
                            left_on='movie_title', right_on='name', how='inner')

# Visualizar los primeros registros de las fusiones para verificar
merged_voice_actors_head = merged_voice_actors.head()
merged_directors_head = merged_directors.head()

merged_voice_actors_head, merged_directors_head

# Fusionar todos los conjuntos de datos en uno solo
# Primero, fusionamos 'merged_voice_actors' con 'merged_directors'
merged_all = pd.merge(merged_voice_actors, merged_directors, 
                      on=['movie_title', 'release_date', 'genre', 'MPAA_rating', 
                          'total_gross', 'inflation_adjusted_gross'], 
                      how='outer')

# Eliminar columnas duplicadas que resultan de la fusión (como 'movie' y 'name' que son duplicados de 'movie_title')
merged_all.drop(columns=['movie', 'name'], inplace=True)

# Mostrar las primeras filas del conjunto de datos fusionado
merged_all_head = merged_all.head()
merged_all_head, merged_all.shape

# Limpieza de datos para prepararlos para el modelado de regresión

# 1. Tratar con valores faltantes
# Vamos a rellenar los valores faltantes en columnas de texto con 'Desconocido'
# y para las columnas numéricas, utilizaremos la mediana o un valor representativo
merged_all.fillna({'genre': 'Desconocido', 'MPAA_rating': 'Desconocido', 
                   'character': 'Desconocido', 'voice-actor': 'Desconocido', 
                   'director': 'Desconocido'}, inplace=True)

# 2. Normalizar y transformar datos
# Convertir las columnas 'total_gross' y 'inflation_adjusted_gross' de formato cadena a numérico
# Eliminamos los caracteres no numéricos como '$' y comas, y luego convertimos a float
merged_all['total_gross'] = merged_all['total_gross'].replace('[\$,]', '', regex=True).astype(float)
merged_all['inflation_adjusted_gross'] = merged_all['inflation_adjusted_gross'].replace('[\$,]', '', regex=True).astype(float)

# 3. Eliminación de duplicados
# Eliminar posibles filas duplicadas
merged_all.drop_duplicates(inplace=True)

# 4. Verificación de consistencia
# Convertir 'release_date' a formato de fecha
merged_all['release_date'] = pd.to_datetime(merged_all['release_date'], errors='coerce')

# Mostrar las primeras filas después de la limpieza y el resumen del conjunto de datos
cleaned_data_head = merged_all.head()
cleaned_data_summary = merged_all.describe()

cleaned_data_head, cleaned_data_summary

# Construye la ruta relativa del archivo CSV
ruta_csv = "..\\data\\processed\\merged_all_cleaned.csv"

# Guarda el DataFrame en el archivo CSV en la ruta especificada
merged_all.to_csv(ruta_csv, index=False)

#EDA

merged_all.dtypes

import matplotlib.pyplot as plt
import seaborn as sns

# Análisis Exploratorio de Datos (EDA)

# Crear algunas visualizaciones para entender mejor los datos

# 1. Distribución de ingresos totales y ajustados por inflación
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(merged_all['total_gross'], kde=False, bins=30)
plt.title('Distribución de Ingresos Totales')
plt.xlabel('Ingresos Totales')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
sns.histplot(merged_all['inflation_adjusted_gross'], kde=False, bins=30)
plt.title('Distribución de Ingresos Ajustados por Inflación')
plt.xlabel('Ingresos Ajustados por Inflación')
plt.ylabel('Frecuencia')
plt.tight_layout()

# 2. Ingresos totales por género
plt.figure(figsize=(12, 6))
sns.boxplot(x='genre', y='total_gross', data=merged_all)
plt.title('Ingresos Totales por Género')
plt.xlabel('Género')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)

# 3. Ingresos totales a lo largo del tiempo
plt.figure(figsize=(12, 6))
merged_all_sorted_by_date = merged_all.sort_values('release_date')
sns.lineplot(x='release_date', y='total_gross', data=merged_all_sorted_by_date)
plt.title('Ingresos Totales a lo largo del Tiempo')
plt.xlabel('Fecha de Lanzamiento')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)

plt.show()

# Realizar un zoom en los datos entre los años 1990 y 2000 para observar el pico de ingresos

# Filtrar los datos para el rango de años 1990 a 2000
filtered_data = merged_all[(merged_all['release_date'] >= '1990-01-01') & 
                           (merged_all['release_date'] <= '2000-12-31')]

# Gráfico de líneas de ingresos totales a lo largo del tiempo para el periodo 1990-2000
plt.figure(figsize=(12, 6))
sns.lineplot(x='release_date', y='total_gross', data=filtered_data)
plt.title('Ingresos Totales de Películas de Disney (1990-2000)')
plt.xlabel('Fecha de Lanzamiento')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)
plt.show()

# Mostrar las películas y sus ingresos en el período de 1990 a 2000

# Seleccionar las columnas relevantes ('movie_title', 'release_date', 'total_gross')
movies_revenue_1990_2000 = filtered_data[['movie_title', 'release_date', 'total_gross']]

# Ordenar los datos por ingresos totales de forma descendente
movies_revenue_sorted = movies_revenue_1990_2000.sort_values('total_gross', ascending=False)

# Eliminar duplicados para mostrar cada película una sola vez
movies_revenue_unique = movies_revenue_sorted.drop_duplicates(subset=['movie_title'])

movies_revenue_unique.head(10)  # Mostrar las 10 películas con mayores ingresos


def format_to_dollars(df, column_name):
    """
    Formatea los valores numéricos de una columna de un DataFrame a formato de dólares.
    """
    df[column_name] = df[column_name].apply(lambda x: f'${x:,.2f}')
    return df

# Aplicar la función para formatear la columna de ingresos totales
movies_revenue_formatted = format_to_dollars(movies_revenue_unique.copy(), 'total_gross')

# Mostrar los resultados
print(movies_revenue_formatted.head(10))

#Analisis Univariables

# Creando gráficas univariantes para cada columna del DataFrame

# Configurando el tamaño del área de las gráficas
plt.figure(figsize=(20, 20))

# 1. Conteo de películas por género
plt.subplot(3, 3, 1)
sns.countplot(y='genre', data=merged_all, order=merged_all['genre'].value_counts().index)
plt.title('Conteo de Películas por Género')
plt.xlabel('Conteo')
plt.ylabel('Género')

# 2. Conteo de películas por calificación MPAA
plt.subplot(3, 3, 2)
sns.countplot(y='MPAA_rating', data=merged_all, order=merged_all['MPAA_rating'].value_counts().index)
plt.title('Conteo de Películas por Calificación MPAA')
plt.xlabel('Conteo')
plt.ylabel('Calificación MPAA')

# 3. Distribución de ingresos totales
plt.subplot(3, 3, 3)
sns.histplot(merged_all['total_gross'], kde=False)
plt.title('Distribución de Ingresos Totales')
plt.xlabel('Ingresos Totales')
plt.ylabel('Frecuencia')

# 4. Distribución de ingresos ajustados por inflación
plt.subplot(3, 3, 4)
sns.histplot(merged_all['inflation_adjusted_gross'], kde=False)
plt.title('Distribución de Ingresos Ajustados por Inflación')
plt.xlabel('Ingresos Ajustados por Inflación')
plt.ylabel('Frecuencia')

# 5. Conteo de personajes
plt.subplot(3, 3, 5)
character_counts = merged_all['character'].value_counts().head(20)  # Top 20 personajes
sns.barplot(x=character_counts.values, y=character_counts.index)
plt.title('Top 20 Personajes más Frecuentes')
plt.xlabel('Conteo')
plt.ylabel('Personaje')

# 6. Conteo de actores de voz
plt.subplot(3, 3, 6)
voice_actor_counts = merged_all['voice-actor'].value_counts().head(20)  # Top 20 actores de voz
sns.barplot(x=voice_actor_counts.values, y=voice_actor_counts.index)
plt.title('Top 20 Actores de Voz más Frecuentes')
plt.xlabel('Conteo')
plt.ylabel('Actor de Voz')

# 7. Conteo de directores
plt.subplot(3, 3, 7)
director_counts = merged_all['director'].value_counts().head(20)  # Top 20 directores
sns.barplot(x=director_counts.values, y=director_counts.index)
plt.title('Top 20 Directores más Frecuentes')
plt.xlabel('Conteo')
plt.ylabel('Director')

# Ajustar el layout
plt.tight_layout()
plt.show()

# Análisis multivariante considerando el objetivo de determinar qué aspectos de una película contribuyen a su éxito

# Configurar el área de las gráficas
plt.figure(figsize=(20, 20))

# 1. Ingresos totales por género y calificación MPAA
plt.subplot(2, 2, 1)
sns.barplot(x='genre', y='total_gross', hue='MPAA_rating', data=merged_all)
plt.title('Ingresos Totales por Género y Calificación MPAA')
plt.xlabel('Género')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)

# 2. Ingresos ajustados por inflación por género y calificación MPAA
plt.subplot(2, 2, 2)
sns.barplot(x='genre', y='inflation_adjusted_gross', hue='MPAA_rating', data=merged_all)
plt.title('Ingresos Ajustados por Inflación por Género y Calificación MPAA')
plt.xlabel('Género')
plt.ylabel('Ingresos Ajustados por Inflación')
plt.xticks(rotation=45)

# 3. Relación entre ingresos totales y ajustados por inflación
plt.subplot(2, 2, 3)
sns.scatterplot(x='total_gross', y='inflation_adjusted_gross', hue='genre', data=merged_all)
plt.title('Relación entre Ingresos Totales y Ajustados por Inflación')
plt.xlabel('Ingresos Totales')
plt.ylabel('Ingresos Ajustados por Inflación')

# 4. Ingresos totales por director (Top 10 directores)
top_directors = merged_all['director'].value_counts().head(10).index
filtered_directors = merged_all[merged_all['director'].isin(top_directors)]
plt.subplot(2, 2, 4)
sns.barplot(x='director', y='total_gross', data=filtered_directors)
plt.title('Ingresos Totales por Director (Top 10)')
plt.xlabel('Director')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)

# Ajustar el layout
plt.tight_layout()
plt.show()

merged_all.dtypes

import pandas as pd

# Suponiendo que merged_all es tu DataFrame
# Si no lo has cargado previamente, asegúrate de cargar tus datos primero

# Tabla de contingencia entre 'genre' y 'MPAA_rating'
contingency_table_genre_mpaa = pd.crosstab(merged_all['genre'], merged_all['MPAA_rating'])
print("Tabla de contingencia entre 'genre' y 'MPAA_rating':")
print(contingency_table_genre_mpaa)
print()

# Tabla de contingencia entre 'character' y 'voice-actor'
contingency_table_character_voice_actor = pd.crosstab(merged_all['character'], merged_all['voice-actor'])
print("Tabla de contingencia entre 'character' y 'voice-actor':")
print(contingency_table_character_voice_actor)
print()

# Tabla de contingencia entre 'director' y 'genre'
contingency_table_director_genre = pd.crosstab(merged_all['director'], merged_all['genre'])
print("Tabla de contingencia entre 'director' y 'genre':")
print(contingency_table_director_genre)
print()

# Calcular el año de lanzamiento como una característica numérica
merged_all['release_year'] = merged_all['release_date'].dt.year

# Tabla de contingencia entre 'MPAA_rating' y 'release_year' (año de lanzamiento)
contingency_table_mpaa_release_year = pd.crosstab(merged_all['MPAA_rating'], merged_all['release_year'])
print("Tabla de contingencia entre 'MPAA_rating' y 'release_year' (año de lanzamiento):")
print(contingency_table_mpaa_release_year)

import pandas as pd
import matplotlib.pyplot as plt

# Suponiendo que merged_all es tu DataFrame
# Si no lo has cargado previamente, asegúrate de cargar tus datos primero

# Tabla de contingencia entre 'genre' y 'MPAA_rating'
contingency_table_genre_mpaa = pd.crosstab(merged_all['genre'], merged_all['MPAA_rating'])

# Tabla de contingencia entre 'MPAA_rating' y 'release_year' (año de lanzamiento)
contingency_table_mpaa_release_year = pd.crosstab(merged_all['MPAA_rating'], merged_all['release_year'])

# Crear gráfico de barras apiladas para 'genre' y 'MPAA_rating'
contingency_table_genre_mpaa.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Gráfico de Barras Apiladas para 'genre' y 'MPAA_rating'")
plt.xlabel("Género")
plt.ylabel("Número de Películas")
plt.legend(title='MPAA_rating', loc='upper right')
plt.show()

# Crear gráfico de barras apiladas para 'MPAA_rating' y 'release_year'
contingency_table_mpaa_release_year.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Gráfico de Barras Apiladas para 'MPAA_rating' y 'release_year'")
plt.xlabel("MPAA_rating")
plt.ylabel("Número de Películas")
plt.legend(title='Año de Lanzamiento', loc='upper right')
plt.show()

#Correlation matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Suponiendo que merged_all es tu DataFrame
# Si no lo has cargado previamente, asegúrate de cargar tus datos primero

# Seleccionar solo las columnas numéricas
numeric_columns = merged_all.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlation_matrix = numeric_columns.corr()

# Crear una figura y un eje para el gráfico
plt.figure(figsize=(10, 8))

# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Mostrar el gráfico
plt.title('Matriz de Correlación')
plt.show()

#1Dataset Backup for regression

merged_all_r = merged_all.copy()

file_path = '../data/interin/merged_all_r.csv'

# Export the DataFrame to a CSV file
merged_all_r.to_csv(file_path, index=False)

import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

# Load the dataset
file_path = '../data/interin/merged_all_r.csv'  # Update with your file path
merged_all_r = pd.read_csv(file_path)

# Identify the categorical columns
categorical_cols = ['genre', 'MPAA_rating', 'character', 'voice-actor', 'director']  # Update as per your dataset

# Separate the features and the target variable
X = merged_all_r.drop(['inflation_adjusted_gross', 'movie_title', 'release_date'], axis=1)
y = merged_all_r['inflation_adjusted_gross']

# Apply one-hot encoding to categorical columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# Convert the sparse matrix to a dense matrix
X_processed_dense = X_processed.toarray() if sparse.issparse(X_processed) else X_processed

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed_dense, y, test_size=0.25, random_state=42)

# Initialize and fit TPOT Regressor
tpot_regressor = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, scoring='r2')
tpot_regressor.fit(X_train, y_train)

# Evaluate the model
print("R2 score on test data:", tpot_regressor.score(X_test, y_test))

# Export the best pipeline
tpot_regressor.export('../models/best_pipeline_for_regression.py')

import joblib

pipeline = '../models/best_pipeline.py'

# Save the pipeline into a model
joblib.dump(pipeline, ('../models/model_pipeline.pkl'))