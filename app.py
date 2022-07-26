# pip install streamlit fbprophet yfinance plotly
# import packages and alias
import numpy as np # data arrays
import pandas as pd # data structure and data analysis
import matplotlib as plt # data visualization
import matplotlib.pyplot as plt
import datetime as dt # date time
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
#import warnings
#warnings.filterwarnings("ignore")
%matplotlib inline
sns.set_theme()
pd.set_option('display.max_columns', None)
import math

import yfinance as yf

st.image("compunnel.png",width=100)

st.title('Stock Forecast App')

stocks = ('AAPL', 'MSFT', 'UNH','GOOG','AMZN') 
selected_stock = st.selectbox('Select dataset for prediction', stocks)

@st.cache
def load_data(ticker):
    data = yf.download(ticker,period = '5y')
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

AAL_df['Close_first']=AAL_df['Close']-AAL_df['Close'].shift(1)

AAL_df.drop(['Close','Adj Close','Open','High','Low','EMA_0.1','Volume'],axis=1,inplace=True)
AAL_df.dropna(inplace=True)

train=AAL_df.loc[AAL_df.index<'2021-05-01']
test=AAL_df.loc[AAL_df.index>='2021-05-01']

reg=load_model('/content/xgboost.h5')
pred=reg.predict(X_test)

st.write('pred')
