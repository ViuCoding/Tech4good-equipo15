#!/usr/bin/env python
# coding: utf-8

# # ***Tech4Good***
# 
# ---

# In[ ]:





# In[110]:


# Importacion de las librerias
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle


# In[6]:


# Lectura del archivo de precipitacion
df_prec = pd.read_csv('precipitacionsbarcelonadesde1786.csv', sep=',')

# Visualizacion de las 3 primeras filas del archivo
df_prec.head(3)


# In[7]:


# Lectura del archivo de temperatura
df_temp = pd.read_csv('temperaturesbarcelonadesde1780.csv', sep=',')

# Visualizacion de las 3 primeras filas del archivo
df_temp.head(3)


# In[8]:


#Eljo una paleta de colores predeterminada
color = 'PRGn' # set palette color (from PALETTE_COLOR)
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

# In[9]:


# Tamaño del archivo
print('El archivo de precipitacion tiene ' + str(df_prec.shape[0]) + ' filas y ' + str(df_prec.shape[1]) + ' columnas')


# In[10]:


# Identificacion de tipos de variables y el recuento de no nulos
df_prec.info()


# In[11]:


# Visualizacion de las columnas
df_prec.columns


# In[12]:


# Revision de valores misssings o nulos
df_prec.isnull().sum()


# In[13]:


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


# In[14]:


# Describo las variables (analisis mensual)
df_prec.describe()


# In[15]:


# Unpivot the columns except 'Any' using the melt() function
df_precipitacion = df_prec.melt(id_vars=['Any'], var_name='Mes', value_name='Precipitacion')

# Create a new 'Date' column by combining 'Month' and 'Year' as a string
df_precipitacion['Fecha'] =  pd.to_datetime(df_precipitacion['Mes'].astype(str) + ' ' + df_precipitacion['Any'].astype(str), format='%m %Y')


# Display the unpivoted DataFrame
print(df_precipitacion)


# In[16]:


# Describo las variables (analisis anual)
df_precipitacion.describe()


# ## *Temperatura*
# 
# 
# ---

# In[17]:


# Tamaño del archivo
print('El archivo de temperatura tiene ' + str(df_temp.shape[0]) + ' filas y ' + str(df_temp.shape[1]) + ' columnas')


# In[18]:


# Identificacion de tipos de variables y el recuento de no nulos
df_temp.info()


# In[19]:


# Visualizacion de las columnas
df_temp.columns


# In[20]:


# Revision de valores misssings o nulos
df_temp.isnull().sum()


# In[21]:


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


# In[22]:


# Describo las variables (analisis mensual)
df_temp.describe()


# In[23]:


# Unpivot the columns except 'Any' using the melt() function
df_temperatura = df_temp.melt(id_vars=['Any'], var_name='Mes', value_name='Temperatura')

# Create a new 'Date' column by combining 'Month' and 'Year' as a string
df_temperatura['Fecha'] =  pd.to_datetime(df_temperatura['Mes'].astype(str) + ' ' + df_temperatura['Any'].astype(str), format='%m %Y')

# Display the unpivoted DataFrame
print(df_temperatura)


# ## *Dataset completo*

# In[24]:


# Nuevo dataset con las columnas 'Fecha', 'Any', 'Mes', 'Precipitacion' y 'Temperatura'
df = df_precipitacion[['Fecha', 'Any', 'Mes', 'Precipitacion']].merge(df_temperatura[['Fecha', 'Temperatura']], on='Fecha')

# Imprimir el nuevo dataset
print(df)


# In[25]:


df = df.sort_values('Fecha')


# In[26]:


import numpy as np
from scipy import stats

def calculate_spi(precipitation_values, historical_mean, historical_std):
    # Calculo el SPI
    spi = (precipitation_values - historical_mean) / historical_std

    return spi

# Initialize the SPI column
df['SPI'] = np.nan

# Iterate over each year
for year in df['Any'].unique():
    # Filter the dataset for the specific year and the last 30 years
    last_30_years_df = df[(df['Any'] >= (year - 30)) & (df['Any'] <= year)]

    # Extract the 'Precipitacion' column from the filtered DataFrame
    precipitation_data = last_30_years_df['Precipitacion']

    # Calculate the historical mean and standard deviation from the filtered dataset
    historical_mean = precipitation_data.mean()
    historical_std = precipitation_data.std()

    # Calculate the SPI for the specific year
    df.loc[df['Any'] == year, 'SPI'] = calculate_spi(df.loc[df['Any'] == year, 'Precipitacion'], historical_mean, historical_std)

print(df)


# In[27]:


from scipy import stats

def calculate_spi(precipitation_values, historical_mean, historical_std):
    # Calculate the SPI
    spi = (precipitation_values - historical_mean) / historical_std

    return spi

# Extract the 'Precipitacion' column from your DataFrame
precipitation_data = df['Precipitacion']

# Calculate the historical mean and standard deviation from the dataset
historical_mean = np.mean(precipitation_data)
historical_std = np.std(precipitation_data)

# Calculate the SPI
df['SPI Historico'] = calculate_spi(precipitation_data, historical_mean, historical_std)


# In[28]:


import numpy as np
from scipy import stats

def calculate_spei(precipitation_values, temperature_values, historical_mean_p, historical_std_p, historical_mean_t, historical_std_t):
    # Calculate the SPI for precipitation
    spi_precipitation = (precipitation_values - historical_mean_p) / historical_std_p

    # Calculate the SPI for temperature
    spi_temperature = (temperature_values - historical_mean_t) / historical_std_t

    # Calculate the SPEI as the difference between SPI for precipitation and SPI for temperature
    spei = spi_precipitation - spi_temperature

    return spei

# Initialize the SPEI column
df['SPEI'] = np.nan

# Iterate over each year
for year in df['Any'].unique():
    # Filter the dataset for the specific year and the last 30 years
    last_30_years_df = df[(df['Any'] >= (year - 30)) & (df['Any'] <= year)]

    # Extract the 'Precipitacion' and 'Temperatura' columns from the filtered DataFrame
    precipitation_data = last_30_years_df['Precipitacion']
    temperature_data = last_30_years_df['Temperatura']

    # Calculate the historical mean and standard deviation for precipitation
    historical_mean_p = precipitation_data.mean()
    historical_std_p = precipitation_data.std()

    # Calculate the historical mean and standard deviation for temperature
    historical_mean_t = temperature_data.mean()
    historical_std_t = temperature_data.std()

    # Calculate the SPEI for the specific year
    df.loc[df['Any'] == year, 'SPEI'] = calculate_spei(df.loc[df['Any'] == year, 'Precipitacion'], df.loc[df['Any'] == year, 'Temperatura'], historical_mean_p, historical_std_p, historical_mean_t, historical_std_t)


# In[29]:


from scipy import stats

def calculate_spei(precipitation_values, temperature_values, historical_mean_p, historical_std_p, historical_mean_t, historical_std_t):
    # Calculate the SPI for precipitation
    spi_precipitation = (precipitation_values - historical_mean_p) / historical_std_p

    # Calculate the SPI for temperature
    spi_temperature = (temperature_values - historical_mean_t) / historical_std_t

    # Calculate the SPEI as the difference between SPI for precipitation and SPI for temperature
    spei = spi_precipitation - spi_temperature

    return spei

# Extract the 'Precipitacion' and 'Temperatura' columns from your DataFrame
precipitation_data = df['Precipitacion']
temperature_data = df['Temperatura']

# Calculate the historical mean and standard deviation for precipitation
historical_mean_p = np.mean(precipitation_data)
historical_std_p = np.std(precipitation_data)

# Calculate the historical mean and standard deviation for temperature
historical_mean_t = np.mean(temperature_data)
historical_std_t = np.std(temperature_data)

# Calculate the SPEI
df['SPEI_Historico'] = calculate_spei(precipitation_data, temperature_data, historical_mean_p, historical_std_p, historical_mean_t, historical_std_t)


# In[30]:


# Export the dataset to a JSON file
df.to_json('dataset.json', orient='records')


# In[31]:


# Calcular tendencia de la precipitación a lo largo de los años
tendencia_precip = np.polyfit(df['Any'], df['Precipitacion'], 1)
df['Tendencia Precipitacion'] = tendencia_precip[0] * df['Any'] + tendencia_precip[1]

# Calcular variabilidad interanual y estacional
df['Var Precip Interanual'] = df.groupby('Any')['Precipitacion'].transform(lambda x: x.std())
df['Var Precip Estacional'] = df.groupby('Mes')['Precipitacion'].transform(lambda x: x.std())


# In[32]:


# Calcular tendencia de la temperatura a lo largo de los años
tendencia_temp = np.polyfit(df['Any'], df['Temperatura'], 1)
df['Tendencia Temperatura'] = tendencia_temp[0] * df['Any'] + tendencia_temp[1]

# Calcular variabilidad interanual y estacional
df['Var Temp Interanual'] = df.groupby('Any')['Temperatura'].transform(lambda x: x.std())
df['Var Temp Estacional'] = df.groupby('Mes')['Temperatura'].transform(lambda x: x.std())


# In[33]:


# Veo los valores del dataset
df.tail(10)


# In[69]:


df[df['Any'] == 2022]


# In[34]:


# Analiso los datos
df.describe()


# In[71]:


# Acorto el dataset
max_fecha = df['Fecha'].max()
fecha_limite = max_fecha - pd.DateOffset(years=30)

df_fecha = df[df['Fecha'] >= fecha_limite]

print(df_fecha.shape)


# # usando todos los años

# In[72]:


df1=df_fecha[["Fecha","SPI Historico"]]
df1


# In[73]:


df1.set_index('Fecha', inplace=True)


# In[74]:


WINDOW_SIZE = 12
HORIZON = 1
df_windowed = df1.copy()
list_cols = list(df1.columns)
# Añadimos del valor -1 hasta el último de la WINDOW_SIZE (-365).
for i in range(1, WINDOW_SIZE + HORIZON):
    print(list_cols[0] + "-" + str(i))
    list_cols.append(list_cols[0] + "-" + str(i))
    df_windowed = pd.concat([df_windowed, df1.shift(i)], axis=1)

df_windowed.head()


# In[75]:


list_cols


# In[76]:


list_cols = ["Target"] + list_cols
list_cols.pop(len(list_cols)-1)
df_windowed.columns = list_cols
df_windowed.head(10)


# In[77]:


df_windowed.dropna(axis=0, inplace=True)


# In[78]:


len(df_windowed)


# In[79]:


X = df_windowed.drop("Target", axis=1)
y = df_windowed["Target"]


# In[82]:


SPLIT_TIME1 =  12*27+1
X_train, X_test = X.iloc[:SPLIT_TIME1], X.iloc[SPLIT_TIME1:]
y_train, y_test = y.iloc[:SPLIT_TIME1], y.iloc[SPLIT_TIME1:]


# In[100]:


y_test


# In[86]:


X_test


# In[89]:


regressor = XGBRegressor(random_state=123)
results = regressor.fit(X_train, y_train)


# In[91]:


y_pred = regressor.predict(X_test)


# In[92]:


y_pred


# In[102]:


# forecast = pd.DataFrame(y_pred,columns=["Target"],
#     index=range(y_test.first_valid_index(), y_test.last_valid_index() + 1)
# )
forecast = pd.DataFrame({'Target': y_pred}, index=y_test.index)


# In[113]:


forecast


# In[104]:


plt.plot(y_test, label="Real")
plt.plot(forecast, "--", label="Pred")

plt.title("Test - Comparativa valor Real vs Predicho", fontsize=16)
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.legend()


# In[109]:


print("Error MSE: ", mean_squared_error(y_test, forecast))
print("Error MAE: ", mean_absolute_error(y_test, forecast))


# In[112]:


pickle.dump(regressor, open("XGBoost_SPI.sav", "wb"))


# In[ ]:


new_model = pickle.load(open("XGBoost_Forecast_SerieSintetica.sav", "rb"))
y_pred = new_model.predict(X_test)

