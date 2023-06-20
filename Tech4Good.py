#!/usr/bin/env python
# coding: utf-8

# # ***Tech4Good***
# 
# ---

# In[174]:


get_ipython().system('pip install pmdarima')


# In[175]:


# Importacion de las librerias

import pandas as pd #para almacenar y manipular datos como df
import matplotlib.pylab as plt #para crear visualizaciones
import numpy as np #para operaciones matemáticas y manipular matrices
import seaborn as sns
import os #para interactuar con el sistema operativo

from sklearn.model_selection import train_test_split #para dividir conjuntos de dato de entrenamiento y test
from sklearn.linear_model import LinearRegression #para modelos de regresión lineal
from sklearn.metrics import mean_squared_error #para medir el rendimiento de un modelo (enfatiza los errores más grandes y puede verse afectado por valores atípicos)
from sklearn.metrics import mean_absolute_error #para medir el rendimiento de un modelo (trata todos los errores por igual y es más resistente a los valores atípicos)
from sklearn.naive_bayes import GaussianNB #para modelo de Naive Bayes
import statsmodels.api as sm #para analisis estadisticos y modelado
from scipy.stats import kendalltau #para realizar la prueba de correlación de Kendall
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict

from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.ensemble import GradientBoostingRegressor
import pickle


# In[176]:


# Lectura del archivo de precipitacion
df_prec = pd.read_csv('precipitacionsbarcelonadesde1786.csv', sep=',')

# Visualizacion de las 3 primeras filas del archivo
df_prec.head(3)


# In[177]:


# Lectura del archivo de temperatura
df_temp = pd.read_csv('temperaturesbarcelonadesde1780.csv', sep=',')

# Visualizacion de las 3 primeras filas del archivo
df_temp.head(3)


# In[178]:


#Eljo una paleta de colores predeterminada
color = 'PRGn' # set palette color (from PALETTE_COLOR)
color_dist = ['#385170', '#82C4C3', '#DBE9B7', '#ECB390', '#CA8A8B', '#B97A95', '#716F81']
sns.set_palette(color) # set palette


# # ***Data Wrangling***
# *Información de la base de datos*
# 
# ---

# ## *Precipitacion*
# 
# 
# ---
# 

# In[179]:


# Tamaño del archivo
print('El archivo de precipitacion tiene ' + str(df_prec.shape[0]) + ' filas y ' + str(df_prec.shape[1]) + ' columnas')


# In[180]:


# Identificacion de tipos de variables y el recuento de no nulos
df_prec.info()


# In[181]:


# Visualizacion de las columnas
df_prec.columns


# In[182]:


# Revision de valores misssings o nulos
df_prec.isnull().sum()


# In[183]:


# Diccionario con el numero de mes
nombre_mes_p = {
    'Precip_Acum_Gener': 1,
    'Precip_Acum_Febrer': 2,
    'Precip_Acum_Marc': 3,
    'Precip_Acum_Abril': 4,
    'Precip_Acum_Maig': 5,
    'Precip_Acum_Juny': 6,
    'Precip_Acum_Juliol': 7,
    'Precip_Acum_Agost': 8,
    'Precip_Acum_Setembre': 9,
    'Precip_Acum_Octubre': 10,
    'Precip_Acum_Novembre': 11,
    'Precip_Acum_Desembre': 12
}

# Renompbro las columnas
df_prec = df_prec.rename(columns = nombre_mes_p)


# In[184]:


# Describo las variables (analisis mensual)
df_prec.describe()


# In[185]:


# Unpivot the columns except 'Any' using the melt() function
df_precipitacion = df_prec.melt(id_vars=['Any'], var_name='Mes', value_name='Precipitacion')

# Create a new 'Date' column by combining 'Month' and 'Year' as a string
df_precipitacion['Fecha'] =  pd.to_datetime(df_precipitacion['Mes'].astype(str) + ' ' + df_precipitacion['Any'].astype(str), format='%m %Y')


# Display the unpivoted DataFrame
print(df_precipitacion)


# In[186]:


# Describo las variables (analisis anual)
df_precipitacion.describe()


# ## *Temperatura*
# 
# 
# ---

# In[187]:


# Tamaño del archivo
print('El archivo de temperatura tiene ' + str(df_temp.shape[0]) + ' filas y ' + str(df_temp.shape[1]) + ' columnas')


# In[188]:


# Identificacion de tipos de variables y el recuento de no nulos
df_temp.info()


# In[189]:


# Visualizacion de las columnas
df_temp.columns


# In[190]:


# Revision de valores misssings o nulos
df_temp.isnull().sum()


# In[191]:


# Diccionario con el numero de mes
nombre_mes_t = {
    'Temp_Mitjana_Gener': 1,
    'Temp_Mitjana_Febrer': 2,
    'Temp_Mitjana_Marc': 3,
    'Temp_Mitjana_Abril': 4,
    'Temp_Mitjana_Maig': 5,
    'Temp_Mitjana_Juny': 6,
    'Temp_Mitjana_Juliol': 7,
    'Temp_Mitjana_Agost': 8,
    'Temp_Mitjana_Setembre': 9,
    'Temp_Mitjana_Octubre': 10,
    'Temp_Mitjana_Novembre': 11,
    'Temp_Mitjana_Desembre': 12
}

# Renombro las columnas
df_temp = df_temp.rename(columns = nombre_mes_t)


# In[192]:


# Describo las variables (analisis mensual)
df_temp.describe()


# In[193]:


# Unpivot the columns except 'Any' using the melt() function
df_temperatura = df_temp.melt(id_vars=['Any'], var_name='Mes', value_name='Temperatura')

# Create a new 'Date' column by combining 'Month' and 'Year' as a string
df_temperatura['Fecha'] =  pd.to_datetime(df_temperatura['Mes'].astype(str) + ' ' + df_temperatura['Any'].astype(str), format='%m %Y')

# Display the unpivoted DataFrame
print(df_temperatura)


# ## *Dataset completo*

# In[194]:


# Nuevo dataset con las columnas 'Fecha', 'Any', 'Mes', 'Precipitacion' y 'Temperatura'
df = df_precipitacion[['Fecha', 'Any', 'Mes', 'Precipitacion']].merge(df_temperatura[['Fecha', 'Temperatura']], on='Fecha')

# Imprimir el nuevo dataset
print(df)


# In[195]:


df = df.sort_values('Fecha')


# In[196]:


import numpy as np
from scipy import stats

def calculate_spi(precipitation_values, historical_mean, historical_std):
    # Calculo el SPI
    spi = (precipitation_values - historical_mean) / historical_std

    return spi

    # Creo la columna de SPI
    df['SPI'] = np.nan

    # Calculo el historico tomando los ultimos 30 años
    df_30_years = df[(df['Any'] >= (year - 30)) & (df['Any'] <= year)]

    # Extract the 'Precipitacion' column from the filtered DataFrame
    precipitation_data = df_30_years['Precipitacion']

    # Calculo la media historica y el desvio std de la precipitacion
    historical_mean = precipitation_data.mean()
    historical_std = precipitation_data.std()

    # Calculo el SPI
    df.loc[df['Any'] == year, 'SPI'] = calculate_spi(df.loc[df['Any'] == year, 'Precipitacion'], historical_mean, historical_std)


# In[197]:


from scipy import stats

def calculate_spi(precipitation_values, historical_mean, historical_std):
    # Calculate the SPI
    spi = (precipitation_values - historical_mean) / historical_std

    return spi

# Extraraigo la columna Precipitacion del df
precipitation_data = df['Precipitacion']

# Calculo la media historica y el desvio std de la precipitacion
historical_mean = np.mean(precipitation_data)
historical_std = np.std(precipitation_data)

# Calculo el SPI
df['SPI Historico'] = calculate_spi(precipitation_data, historical_mean, historical_std)


# In[198]:


import numpy as np
from scipy import stats

def calculate_spei(precipitation_values, temperature_values, historical_mean_p, historical_std_p, historical_mean_t, historical_std_t):
    # Calculo el SPI de precipitacion
    spi_precipitation = (precipitation_values - historical_mean_p) / historical_std_p

    # Calculato el SPI de temperatura
    spi_temperature = (temperature_values - historical_mean_t) / historical_std_t

    # Calculate el SPEI como la diferencia de SPIs
    spei = spi_precipitation - spi_temperature

    return spei

    # Creo la columna
    df['SPEI'] = np.nan

    # Calculo el historico tomando los ultimos 30 años
    df_30_years = df[(df['Any'] >= (year - 30)) & (df['Any'] <= year)]

    # Extraio los datos de precipitacion y temperatura del df de 30 años
    precipitation_data = df_30_years['Precipitacion']
    temperature_data = df_30_years['Temperatura']

    # Calculo el historico y la desviacion std de la precipitacion
    historical_mean_p = precipitation_data.mean()
    historical_std_p = precipitation_data.std()

    # Calculo el historico y la desviacion std de la temperatura
    historical_mean_t = temperature_data.mean()
    historical_std_t = temperature_data.std()

    # Calculo el SPEI
    df.loc[df['Any'] == year, 'SPEI'] = calculate_spei(df.loc[df['Any'] == year, 'Precipitacion'], df.loc[df['Any'] == year, 'Temperatura'], historical_mean_p, historical_std_p, historical_mean_t, historical_std_t)


# In[199]:


from scipy import stats

def calculate_spei(precipitation_values, temperature_values, historical_mean_p, historical_std_p, historical_mean_t, historical_std_t):
    # Calculo el SPI de precipitacion
    spi_precipitation = (precipitation_values - historical_mean_p) / historical_std_p

    # Calculato el SPI de temperatura
    spi_temperature = (temperature_values - historical_mean_t) / historical_std_t

    # Calculate el SPEI como la diferencia de SPIs
    spei = spi_precipitation - spi_temperature

    return spei

# Extraio los datos de precipitacion y temperatura
precipitation_data = df['Precipitacion']
temperature_data = df['Temperatura']

# Calculo el historico y la desviacion std de la precipitacion
historical_mean_p = np.mean(precipitation_data)
historical_std_p = np.std(precipitation_data)

# Calculo el historico y la desviacion std de la temperatura
historical_mean_t = np.mean(temperature_data)
historical_std_t = np.std(temperature_data)

# Calculate the SPEI
df['SPEI_Historico'] = calculate_spei(precipitation_data, temperature_data, historical_mean_p, historical_std_p, historical_mean_t, historical_std_t)


# In[200]:


# Export the dataset to a JSON file
df.to_json('dataset.json', orient='records')


# In[201]:


# Calcular tendencia de la precipitación a lo largo de los años
tendencia_precip = np.polyfit(df['Any'], df['Precipitacion'], 1)
df['Tendencia Precipitacion'] = tendencia_precip[0] * df['Any'] + tendencia_precip[1]

# Calcular variabilidad interanual y estacional
df['Var Precip Interanual'] = df.groupby('Any')['Precipitacion'].transform(lambda x: x.std())
df['Var Precip Estacional'] = df.groupby('Mes')['Precipitacion'].transform(lambda x: x.std())


# In[202]:


# Calcular tendencia de la temperatura a lo largo de los años
tendencia_temp = np.polyfit(df['Any'], df['Temperatura'], 1)
df['Tendencia Temperatura'] = tendencia_temp[0] * df['Any'] + tendencia_temp[1]

# Calcular variabilidad interanual y estacional
df['Var Temp Interanual'] = df.groupby('Any')['Temperatura'].transform(lambda x: x.std())
df['Var Temp Estacional'] = df.groupby('Mes')['Temperatura'].transform(lambda x: x.std())


# In[203]:


# Veo los valores del dataset
df.tail(10)


# In[204]:


# Analiso los datos
df.describe()


# In[205]:


# Acorto el dataset
max_fecha = df['Fecha'].max()
fecha_limite = max_fecha - pd.DateOffset(years=200)

df_fecha = df[df['Fecha'] >= fecha_limite]

print(df_fecha.shape)


# # ***Data Analysis***
# *Análisis de la base de datos*
# 
# ---

# In[206]:


import seaborn as sns
import matplotlib.pyplot as plt

# Tamaño del grafico
plt.figure(figsize=(15, 12))

# Grafico de precipitacion por año
plt.subplot(2, 1, 1)
sns.lineplot(x='Any', y='Precipitacion', data=df, color=color_dist[1])
plt.title('Precipitation by Year')
plt.xlabel('Year')
plt.ylabel('Precipitation')

# Linea de tendencia de precipitacion
sns.regplot(x='Any', y='Precipitacion', data=df, scatter=False, color=color_dist[0])

# Grafico de temperatura por año
plt.subplot(2, 1, 2)
sns.lineplot(x='Any', y='Temperatura', data=df, color=color_dist[2])
plt.title('Temperature by Year')
plt.xlabel('Year')
plt.ylabel('Temperature')

# Linea de tendencia de temperatura
sns.regplot(x='Any', y='Temperatura', data=df, scatter=False, color=color_dist[0])

# Ajusto al layout
plt.tight_layout()

# Muestro el grafico
plt.show()


# Entre 1812 y 1825, la ciudad de Barcelona tuvo que ser abastecida de alimentos desde el extranjero, incluso desde puertos del Báltico, debido a una acumulación de temporadas de sequía extrema que diezmaron las cosechas de las zonas más próximas. Con menos de 300 litros por metro cuadrado, la mitad del promedio anual, los cereales no podían prosperar.

# In[207]:


import seaborn as sns
import matplotlib.pyplot as plt

# Filtro los datos a partir del 2000
df_2000 = df[df['Any'] >= 2000]

# Tamaño del grafico
plt.figure(figsize=(15, 12))

# Grafico de precipitacion
plt.subplot(2, 1, 1)
sns.lineplot(x='Fecha', y='Precipitacion', data = df_2000, color=color_dist[1])
plt.title('Precipitacion mensual (desde 2000)')
plt.xlabel('Fecha')
plt.ylabel('Precipitacion')

# Grafico de temperatura
plt.subplot(2, 1, 2)
sns.lineplot(x='Fecha', y='Temperatura', data = df_2000, color=color_dist[2])
plt.title('Temperatura mensual (desde 2000)')
plt.xlabel('Fecha')
plt.ylabel('Precipitacion')

# Customize the x-axis tick labels
plt.xticks(rotation=45)

# Ajuste de layout
plt.tight_layout()

# Show the plot
plt.show()


# In[208]:


# Create subplots for precipitation and temperature boxplots
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Boxplot de precipitacion
sns.boxplot(x='Any', y='Precipitacion', data=df_2000, ax=axes[0])
axes[0].set_title('Boxplot Precipitacion desde 2000')
axes[0].set_xlabel('Año')
axes[0].set_ylabel('Precipitacion (mm)')

# Boxplot de temperatura
sns.boxplot(x='Any', y='Temperatura', data=df_2000, ax=axes[1])
axes[1].set_title('Boxplot Temperatura desde 2000')
axes[1].set_xlabel('Año')
axes[1].set_ylabel('Temperatura (°C)')

# Ajuste de layout
plt.tight_layout()

# Show the plot
plt.show()


# # ***Machine learning***
# *Modelos predictivos temporales*
# 
# ---

# In[209]:


# Utilizo la fecha como indice y tomo los ultimos 30 años

df.set_index('Fecha', inplace=True)
max_fecha = df.index.max()
fecha_limite = max_fecha - pd.DateOffset(years=30)

df_fecha = df[df.index > fecha_limite]

print(df_fecha.shape)


# ## *SARIMAX*

# Comprobación estacionariedad datos
# 
# 

# In[210]:


# Defino un nuevo dataframe con la columna de SPI
df1 = df_fecha[["SPI Historico"]]
df1


# In[211]:


from statsmodels.tsa.stattools import adfuller

# Realizar prueba de Dickey-Fuller Aumentada en la serie de tiempo
result = adfuller(df1['SPI Historico'])

# Extraer el valor p de la prueba
p_value = result[1]

# Comparar el valor p con un umbral (por ejemplo, 0.05) para determinar la estacionariedad
if p_value <= 0.05:
    print("La serie es estacionaria")
else:
    print("La serie no es estacionaria")


# In[212]:


# Visualizar los datos en una serie temporal
df1['SPI Historico'].plot(figsize=(12, 6))
plt.xlabel('Fecha')
plt.ylabel('SPI Historico')
plt.title('Serie temporal SPI')
plt.show()

# Descomposición de la serie temporal: en sus componentes de tendencia, estacionalidad y residuo
decomposition = sm.tsa.seasonal_decompose(df1['SPI Historico'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualizar los componentes de la descomposición
#En el primer subgráfico se muestra la serie temporal original de los casos.
#En el segundo subgráfico se muestra la componente de tendencia obtenida de la descomposición.
#En el tercer subgráfico se muestra la componente de estacionalidad.
#En el cuarto subgráfico se muestra el residuo de la descomposición.
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df1['SPI Historico'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Tendencia')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Estacionalidad')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuo')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Autocorrelación para la estacionalidad
acf = sm.tsa.stattools.acf(seasonal, fft=True)
plt.stem(acf)
plt.xlabel('Lag')
plt.ylabel('Autocorrelación')
plt.title('Autocorrelación de la estacionalidad')
plt.show()


# Forecast SARIMAX

# In[213]:


# Particion del dataset en train, validacion y test
fin_train = '2013-12-01'
inicio_test = '2014-01-01'
datos_train = df1.loc[: fin_train, :]
datos_test  = df1.loc[inicio_test:, :]

print(f"Fechas train      : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test       : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")


# In[214]:


# Calculo la media y la varianza del test, validacion y train splits
mean_train, mean_test = datos_train.mean(), datos_test.mean()
var_train, var_test = datos_train.var(), datos_test.var()


# In[215]:


print("Chequeo de estacionariedad\n")
print("Diferencias entre media y varianza a lo largo de la serie:")
print(f"Media en train = {mean_train}, media test = {mean_test}")
print(f"Varianza en train = {var_train}, varianza test = {var_test}")


# In[216]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Definir los parámetros del modelo SARIMA (p, d, q, P, D, Q, s)
p = 1  # Orden del componente autoregresivo
d = 1  # Orden de diferenciación
q = 1  # Orden del componente de promedio móvil
P = 1  # Orden del componente autoregresivo estacional
D = 1  # Orden de diferenciación estacional
Q = 1  # Orden del componente de promedio móvil estacional
s = 12  # Longitud de la estacionalidad (en este ejemplo, la serie es mensual)

# Crear el objeto SARIMA y ajustar el modelo a los datos de entrenamiento
model1 = SARIMAX(datos_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit1 = model1.fit()

# Realizar predicciones en los datos de prueba
forecast = model_fit1.forecast(len(datos_test))

# Comparar las predicciones con los valores reales
predictions = forecast

# Calcular los errores de pronóstico
errors = datos_test - predictions


# In[217]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(datos_test, predictions)

# Calcular el Error Absoluto Medio (MAE)
mae = mean_absolute_error(datos_test, predictions)

print("MSE:", mse)
print("MAE:", mae)


# In[218]:


import matplotlib.pyplot as plt

# Crear una figura y ejes para la gráfica
fig, ax = plt.subplots(figsize=(12, 6))

# Graficar los valores reales
ax.plot(datos_test.index, datos_test, label='Valores Reales', color=color_dist[2])

# Graficar las predicciones
ax.plot(datos_test.index, predictions, label='Predicciones', color=color_dist[0])

# Personalizar la gráfica
ax.set_title('Predicciones SARIMA')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor')
ax.legend()

# Rotar las etiquetas del eje x para una mejor visualización
plt.xticks(rotation=45)

# Mostrar la gráfica
plt.show()


# ## *SARIMAX dos variables*

# In[219]:


# Defino un nuevo dataframe con la columna de SPI
df3 = df_fecha[["SPI Historico", "Temperatura"]]
df3


# In[220]:


# Particion del dataset en train, validacion y test
fin_train = '2013-12-01'
inicio_test = '2014-01-01'
datos_train = df3.loc[: fin_train, :]
datos_test  = df3.loc[inicio_test:, :]

print(f"Fechas train      : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test       : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")


# In[221]:


# Calculo la media y la varianza del test, validacion y train splits
mean_train, mean_test = datos_train.mean(), datos_test.mean()
var_train, var_test = datos_train.var(), datos_test.var()


# In[222]:


print("Chequeo de estacionariedad\n")
print("Diferencias entre media y varianza a lo largo de la serie:")
print(f"Media en train = {mean_train}, media test = {mean_test}")
print(f"Varianza en train = {var_train}, varianza test = {var_test}")


# In[223]:


# Particion del dataset en train, validacion y test
fin_train = '2013-12-01'
inicio_test = '2014-01-01'
datos_train = df3.loc[: fin_train, :]
datos_test  = df3.loc[inicio_test:, :]

print(f"Fechas train      : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test       : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")


# In[224]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Definir los parámetros del modelo SARIMAX (p, d, q, P, D, Q, s)
p = 1  # Orden del componente autoregresivo
d = 1  # Orden de diferenciación
q = 1  # Orden del componente de promedio móvil
P = 1  # Orden del componente autoregresivo estacional
D = 1  # Orden de diferenciación estacional
Q = 1  # Orden del componente de promedio móvil estacional
s = 12  # Longitud de la estacionalidad (en este ejemplo, la serie es mensual)

# Crear el objeto SARIMAX y ajustar el modelo a los datos de entrenamiento
model = SARIMAX(datos_train['SPI Historico'], exog=datos_train['Temperatura'], order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit()

# Realizar predicciones en los datos de prueba
forecast = model_fit.forecast(len(datos_test), exog=datos_test['Temperatura'])

# Comparar las predicciones con los valores reales
predictions = forecast

# Calcular los errores de pronóstico
errors = datos_test['SPI Historico'] - predictions


# In[225]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calcular el Error Cuadrado Medio (MSE)
mse = mean_squared_error(datos_test['SPI Historico'], predictions)

# Calcular el Error Absoluto Medio (MAE)
mae = mean_absolute_error(datos_test['SPI Historico'], predictions)

print("Error Cuadrado Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)


# In[226]:


# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(datos_train.index, datos_train['SPI Historico'], label='Train', color=color_dist[2])
plt.plot(datos_test.index, datos_test['SPI Historico'], label='Test', color=color_dist[1])
plt.plot(datos_test.index, predictions, label='Predictions', color=color_dist[0])
plt.xlabel('Fecha')
plt.ylabel('SPI Historico')
plt.title('Regresión SARIMAX: SPI Historico vs. Fecha')
plt.legend()
plt.show()


# In[227]:


# pickle.dump(model1, open("Sarimax.sav", "wb"))


# In[228]:


# new_model = pickle.load(open("Sarimax.sav", "rb"))
# start_index = len(df1)
# end_index = start_index + 24
# forecast_2_years = new_model.predict(params,start=start_index, end=end_index, dynamic=False)


# In[234]:


df1


# In[241]:


#entrenamos sarimax con todos los datos
p = 1  # Orden del componente autoregresivo
d = 1  # Orden de diferenciación
q = 1  # Orden del componente de promedio móvil
P = 1  # Orden del componente autoregresivo estacional
D = 1  # Orden de diferenciación estacional
Q = 1  # Orden del componente de promedio móvil estacional
s = 12  # Longitud de la estacionalidad (en este ejemplo, la serie es mensual)
modelSA = SARIMAX(df1, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fitSA = modelSA.fit()

forecast2year = model_fitSA.get_forecast(24)


# In[242]:


predicted_values = forecast2year.predicted_mean  #predecimos  2 años a partir del modelo entrenado con todo el DF


# In[252]:


predicted_values


# In[255]:


ultimo_indice = df1.index[-1]
fechas_siguientes = pd.date_range(ultimo_indice, periods=25, freq='MS')[1:]
nuevos_indices = pd.Index(fechas_siguientes)
nuevos_valores = pd.DataFrame({'SPI Historico': predicted_values[:24]}, index=fechas_siguientes)
df_actualizado = pd.concat([df1, nuevos_valores])


# In[256]:


df_actualizado


# In[275]:


def asignar_texto(valor):
    if -1.5<=valor <-1:
        return 'Sequia Moderada'
    elif valor <-1.5:
        return 'Sequia Extrema'
    elif -1<=valor <=1:
        return 'Normal'
    elif 1<valor <1.5:
        return 'Húmedad Moderada'
    elif valor >1.6:
        return 'Húmedad Extrema'
    else: return 'Normal'


# In[276]:


df_actualizado['Sequia'] = df_actualizado['SPI Historico'].apply(asignar_texto)


# In[277]:


df_actualizado


# In[278]:


df_actualizado.to_json('SPI.json')


# CONCLUSIONES:
# Se realizaron varias modelos, Arima, Sarimax de 1 y 2 variables y un XGBoost (docTech4goodXGBoost.py), el que tuvo mejores resultados fue Sarimax de 2 variables, sin embargo no contabamos con datos de 2023 y 2024 de temperatura asi que utilizamos Sarimax de 1 sola variables para predecir el SPI se creo una columa de "Sequia" en el dataframe a exportar para identificar mejor el KPI.
