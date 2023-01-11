#importing libraries
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#loading dataset

import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet

from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

data=pd.read_csv('/home/vuelogix/Downloads/air quality prediction app/air quality prediction/air quality prediction/data/Air_pollution_final.csv')
data.head()

cities = ['Mumbai', 'Bengaluru','Delhi','Thiruvananthapuram','Kolkata']
data1=data[data['City'].isin(cities)]
data1.head()

outliers = []
def detect_outliers_iqr(data):
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data:
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code

#PM10
PM10_outliers = detect_outliers_iqr(data1['PM10'])
print("Outliers from IQR method: ", PM10_outliers)
# print(len(PM10_outliers))

# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['PM10'], 10)
ninetieth_percentile = np.percentile(data1['PM10'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['PM10'] = np.where(data1['PM10']<tenth_percentile, tenth_percentile, data1['PM10'])
data1['PM10'] = np.where(data1['PM10']>ninetieth_percentile, ninetieth_percentile, data1['PM10'])
# print("Sample:", sample)
print("New array:",data1['PM10'])

#PM2.5
PM25_outliers = detect_outliers_iqr(data1['PM2.5'])
print("Outliers from IQR method: ", PM25_outliers)

# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['PM2.5'], 10)
ninetieth_percentile = np.percentile(data1['PM2.5'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['PM2.5'] = np.where(data1['PM2.5']<tenth_percentile, tenth_percentile, data1['PM2.5'])
data1['PM2.5']= np.where(data1['PM2.5']>ninetieth_percentile, ninetieth_percentile, data1['PM2.5'])
# print("Sample:", sample)
print("New array:",data1['PM2.5'])

#SO2
SO2_outliers = detect_outliers_iqr(data1['SO2'])
print("Outliers from IQR method: ", SO2_outliers)


# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['SO2'], 10)
ninetieth_percentile = np.percentile(data1['SO2'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['SO2'] = np.where(data1['SO2']<tenth_percentile, tenth_percentile, data1['SO2'])
data1['SO2']= np.where(data1['SO2']>ninetieth_percentile, ninetieth_percentile, data1['SO2'])
# print("Sample:", sample)
print("New array:",data1['SO2'])

#CO
CO_outliers = detect_outliers_iqr(data1['CO'])
print("Outliers from IQR method: ", CO_outliers)

# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['CO'], 10)
ninetieth_percentile = np.percentile(data1['CO'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['CO'] = np.where(data1['CO']<tenth_percentile, tenth_percentile, data1['CO'])
data1['CO']= np.where(data1['CO']>ninetieth_percentile, ninetieth_percentile, data1['CO'])
# print("Sample:", sample)
print("New array:",data1['CO'])


#O3
O3_outliers = detect_outliers_iqr(data1['O3'])
print("Outliers from IQR method: ", O3_outliers)

# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['O3'], 10)
ninetieth_percentile = np.percentile(data1['O3'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['O3'] = np.where(data1['O3']<tenth_percentile, tenth_percentile, data1['O3'])
data1['O3']= np.where(data1['O3']>ninetieth_percentile, ninetieth_percentile, data1['O3'])
# print("Sample:", sample)
print("New array:",data1['O3'].head())

#NO2
O3_outliers = detect_outliers_iqr(data1['NO2'])
print("Outliers from IQR method: ", O3_outliers)


# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['NO2'], 10)
ninetieth_percentile = np.percentile(data1['NO2'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['NO2'] = np.where(data1['NO2']<tenth_percentile, tenth_percentile, data1['NO2'])
data1['NO2']= np.where(data1['NO2']>ninetieth_percentile, ninetieth_percentile, data1['NO2'])
# print("Sample:", sample)
print("New array:",data1['NO2'].head())


#AQI
AQI_outliers = detect_outliers_iqr(data1['AQI'])
print("Outliers from IQR method: ", AQI_outliers)

# Computing 10th, 90th percentiles and replacing the outliers
tenth_percentile = np.percentile(data1['AQI'], 10)
ninetieth_percentile = np.percentile(data1['AQI'], 90)
# print(tenth_percentile, ninetieth_percentile)
data1['AQI'] = np.where(data1['AQI']<tenth_percentile, tenth_percentile, data1['AQI'])
data1['AQI']= np.where(data1['AQI']>ninetieth_percentile, ninetieth_percentile, data1['AQI'])
# print("Sample:", sample)
print("New array:",data1['AQI'])

data2=data1.copy()


df = data2[['Date', 'AQI']].dropna()
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

daily_df = df.resample('D').mean()
d_df = daily_df.reset_index().dropna()


d_df.columns = ['ds', 'y']
# fig = plt.figure(facecolor='w', figsize=(20, 6))
# plt.plot(d_df.ds, d_df.y)


m = Prophet()
m.fit(d_df)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# from datetime import datetime, timedelta
# fig1 = m.plot(forecast)#datenow = datetime.now()
# datenow = datetime(2020, 6, 2)
# dateend = datenow + timedelta(days=90)
# datestart = datetime(2018, 6, 2)
# plt.xlim([datestart, dateend])
# plt.title("AQI forecast", fontsize=20)
# plt.xlabel("Day", fontsize=20)
# plt.ylabel("AQI", fontsize=20)
# plt.axvline(datenow, color="k", linestyle=":")
# plt.show()

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-90:])
# fig2 = m.plot_components(forecast)
from fbprophet.diagnostics import cross_validation, performance_metrics
df_cv = cross_validation(m, horizon='90 days')
df_p = performance_metrics(df_cv)
df_p.head(5)

# from fbprophet.plot import plot_cross_validation_metric
# fig3 = plot_cross_validation_metric(df_cv, metric='mape')


import pickle
with open('../air quality prediction/air quality prediction/forecast_model.pkl', 'wb') as fout:
    pickle.dump(m, fout)



