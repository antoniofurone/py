import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from fbprophet import Prophet

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))


url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)



df_tot_casi = df.loc[:,['data','totale_attualmente_positivi']]
#print(df_tot_casi.tail)

tot_casi_prophet = df_tot_casi[['data', 'totale_attualmente_positivi']]
tot_casi_prophet.columns = ['ds', 'y']
#print(tot_casi_prophet.tail)

model_tot_casi = Prophet(interval_width=0.99)
model_tot_casi.fit(tot_casi_prophet)
future_tot_casi = model_tot_casi.make_future_dataframe(periods=10)
forecast_tot_casi = model_tot_casi.predict(future_tot_casi)

#print(forecast_tot_casi.tail())

forecast_y = []
forecast_y_u = []
forecast_y_l = []
real_y = []

for index, row in forecast_tot_casi.iterrows():

    forecast_y.append(row['yhat'])
    forecast_y_l.append(row['yhat_lower'])
    forecast_y_u.append(row['yhat_upper'])

for index, row in tot_casi_prophet.iterrows():
    real_y.append(row['y'])

#print(tot_casi_yhat)


plt.xlabel('Days')
plt.plot(real_y, label='Confirmed', color='orange')
plt.plot(forecast_y, label='Prediction', color='blue')
plt.plot(forecast_y_l, label='Prediction lower', color='green')
plt.plot(forecast_y_u, label='Predicition upper', color='red')
plt.title("Forecast Total Cases")
plt.legend()
plt.show()

# deceduti
df_death = df.loc[:,['data','deceduti']]
death_prophet = df_death[['data', 'deceduti']]
death_prophet.columns = ['ds', 'y']

model_death = Prophet(interval_width=0.99)
model_death.fit(death_prophet)
future_death = model_death.make_future_dataframe(periods=10)
forecast_death = model_death.predict(future_death)

forecast_y = []
forecast_y_u = []
forecast_y_l = []
real_y = []

for index, row in forecast_death.iterrows():

    forecast_y.append(row['yhat'])
    forecast_y_l.append(row['yhat_lower'])
    forecast_y_u.append(row['yhat_upper'])

for index, row in death_prophet.iterrows():
    real_y.append(row['y'])

#print(tot_casi_yhat)


plt.xlabel('Days')
plt.plot(real_y, label='Confirmed', color='orange')
plt.plot(forecast_y, label='Prediction', color='blue')
plt.plot(forecast_y_l, label='Prediction lower', color='green')
plt.plot(forecast_y_u, label='Predicition upper', color='red')
plt.title("Forecast Death")
plt.legend()
plt.show()

# intensiva
df_intensiva = df.loc[:,['data','terapia_intensiva']]
intensiva_prophet = df_intensiva[['data', 'terapia_intensiva']]
intensiva_prophet.columns = ['ds', 'y']

model_intensiva = Prophet(interval_width=0.99)
model_intensiva.fit(intensiva_prophet)
future_intensiva = model_intensiva.make_future_dataframe(periods=10)
forecast_intensiva = model_intensiva.predict(future_intensiva)

forecast_y = []
forecast_y_u = []
forecast_y_l = []
real_y = []

for index, row in forecast_intensiva.iterrows():

    forecast_y.append(row['yhat'])
    forecast_y_l.append(row['yhat_lower'])
    forecast_y_u.append(row['yhat_upper'])

for index, row in intensiva_prophet.iterrows():
    real_y.append(row['y'])

#print(tot_casi_yhat)


plt.xlabel('Days')
plt.plot(real_y, label='Confirmed', color='orange')
plt.plot(forecast_y, label='Prediction', color='blue')
plt.plot(forecast_y_l, label='Prediction lower', color='green')
plt.plot(forecast_y_u, label='Predicition upper', color='red')
plt.title("Forecast Intensive Therapy")
plt.legend()
plt.show()




