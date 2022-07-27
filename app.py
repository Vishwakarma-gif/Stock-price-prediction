import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf
from keras.models import load_model
import streamlit as st
import pandas_datareader as data

st.set_page_config(page_title = "Compunnel digital")
st.image("compunnel.png",width=100)

st.title('Stock Trend Prediction')
stocks = ('AAPL', 'MSFT', 'UNH','GOOG','AMZN') 

user_input = st.selectbox('Select dataset for prediction', stocks)


start = '2017-01-01'

df = data.DataReader(user_input,'yahoo',start)

#Describing Data

st.write(df.describe())

#Visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,label ='Closing Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b',label ='Closing Price')
plt.plot(ma100,'r',label ='MA100')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r',label ='MA100')
plt.plot(ma200, 'g',label ='MA200')
plt.plot(df.Close, 'b',label ='Closing Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load Model

model = load_model('keras_model (1).h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index =True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test , y_test = np.array(x_test), np.array(y_test)

#Making Predictions

y_predicted = model.predict(x_test)

#scaler = scaler.scale_

#scale_factor = 1/scaler[0]
#y_predicted = y_predicted * scale_factor
#y_test = y_test * scale_factor

y_test = y_test.reshape(-1,1)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test)

yp = pd.DataFrame(y_predicted,columns=(['Predicted']))
yt = pd.DataFrame(y_test,columns=(['Actual']))
y = pd.concat([yt,yp],axis = 1)
dd = data.DataReader(user_input,'yahoo',start)
dd = dd[-421:]
d1 = dd.index
#d2 = d1.split('T')
y1 = y.set_index(d1)
st.subheader('Actual Vs Predicted Prices')
st.write(y1.tail(5))

#Final Graph
st.subheader('Actual Vs Predicted Chart')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Metrics')
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
scores = [['Mean Absolute Error', mean_absolute_error(y_test,y_predicted)], ['Mean Squared Error', mean_squared_error(y_test, y_predicted)], ['Root Mean Squared Error', np.sqrt(mean_squared_error(y_test, y_predicted))],['R2 Score', r2_score(y_test, y_predicted)]]
scores_f = pd.DataFrame(scores, columns=['Metrics', 'Scores'])
st.write(scores_f)





