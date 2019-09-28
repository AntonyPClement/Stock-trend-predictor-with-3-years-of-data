# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:34:41 2019

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train=pd.read_csv("ASIANPAINTALLN.csv")
train_set=dataset_train.iloc[0:481,8:9].values
#normalization
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
scaled_trainset=sc.fit_transform(train_set)
#data structure with 80 timesteps and 1 output
xtrain=[]
ytrain=[]
for i in range(80,481):
    xtrain.append(scaled_trainset[i-80:i,0])
    ytrain.append(scaled_trainset[i,0])
xtrain,ytrain=np.array(xtrain),np.array(ytrain)
#reshaping
xtrain=np.reshape(xtrain,(xtrain.shape[0], xtrain.shape[1], 1)) 

#rnn lstm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

reg=Sequential()

reg.add(LSTM(units=256,return_sequences=True,input_shape=(xtrain.shape[1], 1)))
reg.add(Dropout(rate=0.3))
reg.add(LSTM(units=256,return_sequences=True))
reg.add(Dropout(rate=0.3))
reg.add(LSTM(units=256,return_sequences=True))
reg.add(Dropout(rate=0.3))
reg.add(LSTM(units=256))
reg.add(Dropout(rate=0.3))
reg.add(Dense(units=1))

reg.compile(optimizer='adam',loss='mean_squared_error')

reg.fit(xtrain,ytrain, epochs=75, batch_size=16)

#predictions and results
dataset_test=pd.read_csv("ASIANPAINTALLN.csv")
real_stocks=dataset_test.iloc[481:495,8:9].values

#total_dataset=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
#inputs=total_dataset[len(total_dataset)-len(dataset_test)-80:].values
total_dataset=dataset_train['Close Price']
inputs=total_dataset[len(total_dataset)-len(real_stocks)-80:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
xtest=[]
for i in range(80,94):
    xtest.append(inputs[i-80:i,0])
xtest=np.array(xtest)
#reshaping
xtest=np.reshape(xtest,(xtest.shape[0], xtest.shape[1], 1))

predict_stocks=reg.predict(xtest)
predict_stocks=sc.inverse_transform(predict_stocks)

#visualization
plt.plot(real_stocks, color='red', label='real stock prices')
plt.plot(predict_stocks, color='blue', label='predicted stock prices')
plt.title('Close Stock price prediction')
plt.xlabel('time')
plt.ylabel('close stock price')
plt.legend()
plt.show()


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stocks, predict_stocks))
print(rmse)

import pickle
weigh= reg.get_weights();    pklfile= "D:/modelweights.pkl"
try:
    fpkl= open(pklfile, 'wb')    #Python 3     
    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    fpkl.close()
except:
    fpkl= open(pklfile, 'w')    #Python 2      
    pickle.dump(weigh, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
    fpkl.close()