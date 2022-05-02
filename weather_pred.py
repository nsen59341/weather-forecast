### IMPORT LIBRARIES
import warnings 
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from wwo_hist import retrieve_hist_data
import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset
import streamlit as st
import warnings
warnings.simplefilter("ignore")

st.title("Weather Forecasting")
st.write("""
***This App forecast the weather of future date***
""")

selected_date = st.date_input("Pick a date")
# selected_time = st.time_input("Pick a time")
selected_location = st.selectbox("Select a Location", ('Ahmedabad','Bangalore','Delhi','Hydrabad','Kolkata','Mumbai','Pune','Srinagar'))

st.write("Temperature of ", selected_location, " on ", selected_date)

frequency = 3
start_date = '01-FEB-2022'
end_date = '1-MAY-2022'
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
data = data.groupby(data.index.date).mean()

ts = data['tempC']

def test_stationarity(timeseries):
    plt.figure(figsize=(16,9))
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=8,center=False).mean() 
    rolstd = timeseries.rolling(window=8,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    
    #Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value


test_stationarity(ts)

ts_log = np.log(ts)

movingAvg = ts_log.rolling(window=8).mean()
movingStd = ts_log.rolling(window=8).std()

ts_log_mv_diff = ts_log - movingAvg
ts_log_mv_diff.dropna(inplace=True)

test_stationarity(ts_log_mv_diff)

### Now let's try Autoarima

# divide into train and validation set
train = data[:int(0.8*(len(data)))]
test = data[int(0.8*(len(data))):]

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
results = model.fit(train)

forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])

rms = sqrt(mean_squared_error(test,forecast))

date1 = datetime.datetime.today()

date2 = pd.to_datetime(selected_date)
# st.write(date1,' ',date2)
# n = (date2-date1).days+1
# st.write("n ",n)
feauture_data = results.predict(start=date2, n_periods=1)

future_dates = [(date2 + datetime.timedelta(days=x)) for x in range(0,1)]

# st.write(future_dates)

future_datest_data=pd.DataFrame(index=future_dates[0:],columns=['tempC'])

future_datest_data['tempC'] = feauture_data

st.write("{:.2f}".format(feauture_data[0]))

new_data = pd.concat([data[['tempC']],future_datest_data], axis=0)

f, ax = plt.subplots(figsize=(16, 9))
ax = plt.plot(new_data)
st.pyplot(f)