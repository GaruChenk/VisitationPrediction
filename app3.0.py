import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.models
import os
import numpy as np
import sched, time
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from keras.models import Sequential
import joblib   
from gc import callbacks
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
from calendar import month
from datetime import datetime
from tkinter import Scale
import streamlit as st
import pandas as pd

st.title("Stocklands Visitation Prediction Engine")
st.markdown("Version 3.0")

Asset = st.selectbox("Asset", options=('Baldivis', 'Balgowlah', 'Birtinya', 'Bull Creek', 'Burleigh', 'Gladstone', 'Glendale', 'Green Hills', 'Harrisdale', 'Hervey Bay','Merrylands', 'Riverton', 'Rockhampton', 'Wendouree','Wetherill Park'))
n_input = st.number_input("Input Hours", min_value=0, max_value=100)
n_output = st.number_input("Output Hours", min_value=0, max_value=100)
features = st.multiselect("Features For Prediction", default=['feelslike', 'dew','humidity', 'precip', 'precipprob', 'windspeed', 'winddir','sealevelpressure', 'cloudcover', 'visibility', 'severerisk','feelslikemax', 'moonphase', 'feelslikemin', 'precipcover','sunrise_hour', 'sunset_hour'], options=['feelslike', 'dew','humidity', 'precip', 'precipprob', 'windspeed', 'winddir','sealevelpressure', 'cloudcover', 'visibility', 'severerisk','feelslikemax', 'moonphase', 'feelslikemin', 'precipcover','sunrise_hour', 'sunset_hour'])


features.append('Asset')
features.append('Visitation')
features.append('Year')
features.append('Month')

if st.button("Predict Vistations", key=1):
    resultsMAE = {}
    resultsMSE = {}
    resultsR2 = {}

    asset = Asset

    def split_data(df,train_size):
        train_days=math.floor(len(df)*train_size/24)
        train_data,test_data=df.iloc[0:train_days*24],df.iloc[train_days*24:len(df)]
        return train_data,test_data

    def scale_data(train_data, test_data):
            f_columns = df_pca.drop(columns=['Visitation']).columns
            f_transformer = MinMaxScaler()
            t_transformer = MinMaxScaler()
            f_transformer = f_transformer.fit(train_data[f_columns].to_numpy())
            t_transformer = t_transformer.fit(train_data[['Visitation']])
            train_data.loc[:, f_columns] = f_transformer.transform(train_data[f_columns].to_numpy())
            train_data['Visitation'] = t_transformer.transform(train_data[['Visitation']])
            test_data.loc[:, f_columns] = f_transformer.transform(test_data[f_columns].to_numpy())
            test_data['Visitation'] = t_transformer.transform(test_data[['Visitation']])
            return f_transformer, t_transformer, train_data, test_data

    def create_trainable_dataset(dataframe,n_inputs,n_outputs):
            X,Y=list(),list()
            dataframe2 = dataframe.drop(columns=['Visitation'])
            for i in range(len(dataframe)-n_inputs-n_outputs+1):
                    X.append(dataframe2.iloc[i:(i+n_inputs), :])
                    Y.append(dataframe.iloc[i + n_inputs:i + n_inputs + n_outputs, list(dataframe.columns).index('Visitation')])
            return np.array(X), np.array(Y)

    def build_model(hp):
        model = Sequential()
        model.add(Bidirectional(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32),return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]))))
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
        model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
        model.add(Dense(y_train.shape[1], activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
        return model

    def fit_model(model):
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                patience = 10)
        history = model.fit(X_train, y_train, epochs = 1,  
                            validation_split = 0.2, batch_size = 32, 
                            shuffle = False, callbacks = [early_stop])
        return history

    def prediction(model, X_tesa):
        prediction = model.predict(X_tesa)
        prediction = t_trans.inverse_transform(prediction)
        return prediction


    df = pd.read_csv('streamlit_app/data_withna_16thOct_USING_REPURCHASED_DATE.csv', index_col = [0])

    #Deleting Redundant Columns
    del df['preciptype']
    del df['snow']
    del df['snowdepth']
    del df['windgust']
    del df['stations']
    del df['Address']
    del df['sunrise_second']
    del df['sunset_second']
    del df['sunrise_minute']
    del df['sunset_minute']

    #Filling NAs
    df['Visitation'] = df['Visitation'].fillna(0)
    df['sealevelpressure'] = df['sealevelpressure'].ffill()
    df['visibility'] = df['visibility'].ffill()
    df['solarradiation'] = df['solarradiation'].ffill()
    df['solarenergy'] = df['solarenergy'].fillna(0)
    df['uvindex'] = df['uvindex'].ffill()
    df['severerisk'] = df['severerisk'].fillna(0)


    # need to drop logically
    need_drop = ['Time', 'conditions', 'icon', 'description', 'Day', 'Day name', 'uvindex', 'Hour']

    # need to due to colinearity
    need_drop_corr = ['temp', 'solarradiation', 'solarenergy', 'tempmax', 'tempmin']

    target = ['Visitation']
                
    discrete = ['Year', 'Month', 'precipprob', 
                'severerisk', 'sunrise_hour', 'sunset_hour']
                
    continuous = ['feelslike', 'dew', 'precip', 'windspeed', 'winddir', 
                'sealevelpressure', 'cloudcover', 'visibility', 'feelslikemax', 'moonphase', 
                'feelslikemin', 'precipcover', 'winddir','humidity'
                ]

    nominal = ['Asset', 'Day name']

    df = df.drop(need_drop, axis=1)
    df = df.drop(need_drop_corr, axis=1)

    df.sort_values(by=['Date'], inplace=True)
    df.drop(columns=['Date'], inplace=True)

    df = df[df['Asset'] == asset]

    Ordinisecategoricals = ['Asset','Year', 'Month', 'severerisk', 'sunrise_hour', 'sunset_hour', 'precipprob']

    for i in  Ordinisecategoricals:
        df2 = pd.DataFrame()
        df2[i] = df[i].unique()
        df2['Mean'] = df[[i, 'Visitation']].groupby(df[i])['Visitation'].transform('mean').unique()
        df2.sort_values(by=['Mean'], inplace=True)
        df2.reset_index(inplace=True)
        dictionary = df2[[i]].to_dict()
        dictionary = dict([(value, key) for key, value in dictionary[i].items()])
        df.replace({i: dictionary}, inplace=True)

    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    del df['index']

    df = df[features]

    print(df)

    from sklearn.decomposition import PCA
    X = df.drop(columns=['Visitation'])

    NC = 3
    pca = PCA(n_components = NC)
    X_pca = pca.fit_transform(X)


    col = []
    for i in range(NC):
        col.append(f'PC{i}')

    df_pca = pd.DataFrame(X_pca, columns = col)

    df_pca['Visitation'] = df['Visitation']
    train_size=0.8

    train_data, test_data = split_data(df_pca, train_size)
    f_trans, t_trans, s_train_data, s_test_data = scale_data(train_data, test_data)
    X_train, y_train = create_trainable_dataset(s_train_data, n_input, n_output)
    X_test, y_test = create_trainable_dataset(s_test_data, n_input, n_output)

    best_model = keras.models.load_model(f'{asset} model')
    # history_bilstm = fit_model(best_model)
    y_pred1 = prediction(best_model, X_test)

    mae = np.mean(np.abs(y_pred1-t_trans.inverse_transform(y_test)))
    mse = mean_squared_error(y_pred1, t_trans.inverse_transform(y_test))
    r2 = r2_score(y_pred1, t_trans.inverse_transform(y_test))
    resultsMAE.update({asset:mae})
    resultsMSE.update({asset:mse})
    resultsR2.update({asset:r2})

    st.text(f"The predictive Mean Absolute Error for {asset} is {resultsMAE[asset]}")
