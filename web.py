import plotly.express as px
import streamlit as st
import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import date, timedelta
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.callbacks import EarlyStopping
class Model():
    def __init__(self,feats: int):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(feats, 1)))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.callback = EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=3,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
    def fit(self,train_X: np.ndarray,train_y: np.ndarray):
        history = self.model.fit(train_X, train_y, epochs=100,batch_size=len(train_X) // 10, callbacks=[self.callback])
    def forecast(self,X: np.ndarray):
        return self.model.predict(X)
class prepare():
    def __init__(self):
        self.X = []
        self.y = []
        self.forecast_df = None
    def prepare_train(self,df: pd.DataFrame):
        data = pd.DataFrame(df['Close'])
        self.mm = MinMaxScaler()
        x = data['Close'].values
        shape = x.shape
        x_reshape = x.reshape(-1, 1)
        x_scaled = self.mm.fit_transform(x_reshape)
        x_scaled_reshape = np.reshape(x_scaled, shape)
        data['Close'] = x_scaled_reshape
        i = 0
        while (i + 465 < len(data)):
            self.X.append(data['Close'][i:i + 100])
            self.y.append(data['Close'][i + 465])
            i += 1
        self.X = np.asarray(self.X)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
        self.y = np.asarray(self.y)
        return self.X, self.y
    def prepare_forecast(self,df: pd.DataFrame,days: int):
        forecast_df = df[-465:]
        forecast_df.head()
        data = pd.DataFrame(forecast_df['Close'])
        x = data['Close'].values
        shape = x.shape
        x_reshape = x.reshape(-1, 1)
        x_scaled = self.mm.transform(x_reshape)
        x_scaled_reshape = np.reshape(x_scaled, shape)
        data['Close'] = x_scaled_reshape
        self.forecast_df = data
        self.X = []
        i = 0
        while (i<=days and i+100<len(data)):
            self.X.append(data['Close'][i:i + 100])
            i += 1
        self.X = np.asarray(self.X)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
        return self.X
    def transform_forecast(self, forecast_y: np.ndarray):
        data = self.forecast_df
        x = data['Close'].values
        shape = x.shape
        x_reshape = x.reshape(-1, 1)
        x_scaled = self.mm.inverse_transform(x_reshape)
        x_scaled_reshape = np.reshape(x_scaled, shape)
        data['Close'] = x_scaled_reshape
        self.forecast_df = data.copy()
        ds = data.reset_index()
        forecast_y = forecast_y.reshape(-1, 1)
        forecast_y = self.mm.inverse_transform(forecast_y)
        forecast_y = np.reshape(forecast_y, (len(forecast_y),))
        s = ds['Close'][len(ds) - 1] - forecast_y[0]
        forecast_y += s
        new_ds = {
            'Date': [],
            'Forecast': []
        }
        native_ds = {
            'Date': [],
            'Close': []
        }
        for i in range(len(forecast_y)):
            x = date.today() + timedelta(i - 1)
            new_ds['Date'].append(x)
            new_ds['Forecast'].append(forecast_y[i])
            native_ds['Date'].append(x)
            native_ds['Close'].append(forecast_y[i])
        new_ds = pd.DataFrame(new_ds)
        new_ds['Date'] = pd.to_datetime(new_ds['Date'])
        native_ds = pd.DataFrame(native_ds)
        native_ds['Date'] = pd.to_datetime(native_ds['Date'])
        dt = ds['Close']
        dk = native_ds['Close']
        d = pd.concat([ds, native_ds])
        data = d['Close']
        ewm = data.ewm(span=20, adjust=False).mean()
        d['EWM'] = ewm
        d.drop('Close', 1, inplace=True)
        plot_df = pd.concat([ds, new_ds])
        plot_df['EMA'] = d['EWM']
        plot_df.set_index('Date', inplace=True)
        return forecast_y[-1], self.forecast_df, plot_df
class UI():
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Now Predict Your Stock Prices Easily!")
        self.code = st.text_input('Put Your Stock code here:').upper()
        if self.code:
            with st.spinner('Fetching Stock Data...'):
                self.df = self.fetch_data()
            if not(self.df.empty):
                with st.spinner('Computing Stocks'):
                    self.model,self.p = self.compute_model()
                    st.balloons()
                self.tabs(["Stock Details", "Stock Closing Price plots"])
                self.k = self.container('Forecast')
                self.plot_predictions()
            else:
                st.snow()
                st.error("No Stock Code Named \'" + self.code + "\' Found")
    @st.cache(show_spinner=False)
    def fetch_data(self):
        df = pd.DataFrame()
        try:
            df = web.DataReader(self.code, data_source="yahoo", start='2018-01-01', end=date.today())
        except:
            return df
        data = df['Close']
        ema_short = data.ewm(span=20, adjust=False).mean()
        df['EMA'] = ema_short
        return df
    @st.cache(allow_output_mutation=True,show_spinner=False)
    def compute_model(self):
        p = prepare()
        X, y = p.prepare_train(self.df)
        model = Model(feats=X.shape[1])
        model.fit(X, y)
        return model, p
    def tabs(self,l):
        tab1, tab2 = st.tabs(l)
        tab1.header(self.code + ' '+l[0]+':')
        tab1.dataframe(self.df.tail())
        tab2.header(self.code + ' '+l[1]+':')
        fig = px.line(self.df, x=self.df.index, y=['Open', 'Close', 'EMA'], title='Open, closing rate and EMA')
        tab2.plotly_chart(fig, use_container_width=True)
    def container(self,name):
        c = st.container()
        c.header(name+' plot')
        k = c.slider(name+' time', 1, 365)
        return k
    def plot_predictions(self):
        forecast_X = self.p.prepare_forecast(self.df, self.k)
        forecast_y = self.model.forecast(forecast_X)
        forecast_y, forecast_df, plot_df = self.p.transform_forecast(forecast_y)
        fig = px.line(plot_df, x=plot_df.index, y=['Close', 'Forecast', 'EMA'], title='Forecasting')
        st.plotly_chart(fig, use_container_width=True,)
if __name__ == '__main__':
    UI()
