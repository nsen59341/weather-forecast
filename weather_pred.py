### Install libraries

### IMPORT LIBRARIES
import warnings 
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wwo_hist import retrieve_hist_data
import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset
import streamlit as st


st.title("Weather Forecasting")
st.write("""
***This App forecast the weather of future date***
""")

selected_date = st.date_input("Pick a date")
selected_time = st.time_input("Pick a time")
selected_location = st.selectbox("Select a Location", ('Ahmedabad','Bangalore','Delhi','Hydrabad','Kolkata','Mumbai','Pune','Srinagar'))

st.write("Temperature of ", selected_location, " on ", selected_date, "at ", selected_time)

frequency = 3
start_date = '4-APR-2022'
end_date = '4-MAY-2022'
api_key = '47712b8499b545e0b63201153222604'
location_list = [selected_location.lower()]
hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)
csv_file = selected_location.lower()+".csv"
data = pd.read_csv(csv_file)

##### Converting a object to datetime 
data['date_time'] = pd.to_datetime(data['date_time'])

##### We can also drop the location as all thelocatins are same
data = data[['date_time','tempC']]
data = data.set_index('date_time')

ts = data['tempC']

# def test_stationarity(timeseries):
#     plt.figure(figsize=(16,9))
#     #Determing rolling statistics
#     rolmean = timeseries.rolling(window=12,center=False).mean() 
#     rolstd = timeseries.rolling(window=12,center=False).std()

#     #Plot rolling statistics:
#     orig = plt.plot(timeseries, color='blue',label='Original')
#     mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    
#     #Perform Dickey-Fuller test:
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value


# test_stationarity(ts)


# divide into train and validation set
train = data[:int(0.8*(len(data)))]
test = data[int(0.8*(len(data))):]

m1 = auto_arima(train, error_action='ignore', seasonal=True, m=8, suppress_warnings=True, max_p = 2, max_q = 2, start_p = 1, start_q = 1
)
results = m1.fit(train)

forecast = m1.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['tempP'])

# st.write(train)

selected_date_time = pd.to_datetime(str(selected_date)+' '+str(selected_time))
date1 = datetime.datetime.today()
date2 = selected_date_time
# st.write(date1,' ',date2)

m2 = auto_arima(test, error_action='ignore', seasonal=True, m=8, suppress_warnings=True, max_p = 2, max_q = 2, start_p = 1, start_q = 1
)
results2 = m2.fit(test)

len_test = (date2-date1).days+2

# st.write("n ",len_test)

future_dates = [(date1 + datetime.timedelta(days=x)) for x in range(0,len_test)]
future_dates = [datetime.datetime(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:10]), int(str(x)[11:13]) , 0, 0) for x in future_dates]
# st.write(future_dates[2])

forecast2 = m2.predict(n_periods=len_test)
# st.write(forecast2)

forecast2 = pd.DataFrame(forecast2,index = future_dates, columns=['tempP'])
# st.write(forecast2)
# forecast2.index = forecast2.index.apply(lambda x: datetime.datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), int(x[11:13]) , int(x[14:16]), '00'))
# forecast2.index = forecast2.index.astype('str')
# forecast2_index = [datetime.datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), int(x[11:13]) , 0, 0) for x in forecast2.index]

filter_val = datetime.datetime(int(str(selected_date_time)[:4]), int(str(selected_date_time)[5:7]), int(str(selected_date_time)[8:10]), int(str(selected_date_time)[11:13]) , 0, 0)

# st.write("filter_val ",filter_val)
our_val = forecast2.filter(items=[filter_val], axis=0)
our_val = our_val.values
st.write(our_val)

# rms = sqrt(mean_squared_error(test,forecast))

# st.write(date1,' ',date2)
# n = (date2-date1).days+1
# st.write("n ",n)
# feauture_data = model.predict(n_periods=len(test))



# st.write(future_dates)

# future_datest_data=pd.DataFrame(index=future_dates[0:],columns=['tempC'])

# future_datest_data['tempC'] = forecast

st.write("{:.2f}".format(our_val[0][0]))

new_data = pd.concat([data['tempC'],forecast2], axis=0)

f, ax = plt.subplots(figsize=(16, 9))
ax = plt.plot(new_data)
st.pyplot(f)
