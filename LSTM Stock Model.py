#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to predict the closing 
# stock price of a corporation (Apple Inc.)


# In[2]:


# Libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[3]:


tickerSymbol = 'FB'


# In[4]:


df = web.DataReader(tickerSymbol, data_source='yahoo', start = '2012-01-01', end='2020-05-05')
df


# In[5]:


df.shape


# In[6]:


plt.figure(figsize = (16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()


# In[7]:


#DF with Just Close Column
data = df.filter(['Close'])
dataset = data.values #NP array

#Rows to train model on
training_data_length = math.ceil(len(dataset) * .8)


# In[8]:


#Scale Data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[9]:


# Create Training Data Set
# Created Scaled training data set
train_data = scaled_data[0:training_data_length , :]
#Split into x_train and y_train sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
    


# In[10]:


#Convert x_train y_train to np arr
x_train, y_train = np.array(x_train), np.array(y_train)


# In[11]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[12]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[13]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[14]:


model.fit(x_train, y_train, batch_size = 1, epochs = 1)


# In[15]:


#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_length - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_length:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[16]:


#Convert data to np array
x_test = np.array(x_test)


# In[17]:


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[18]:


#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[19]:


#Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[20]:


#Plot the data
train = data[:training_data_length]
valid = data[training_data_length:]
valid['Predictions'] = predictions
#Visualize the Data
plt.figure(figsize = (16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show


# In[21]:


#Show the valid and predicted prices
valid


# In[22]:


#Get the quote
apple_quote = web.DataReader(tickerSymbol, data_source = 'yahoo', start = '2012-01-01', end = '2020-05-04')
#Create a new Dataframe
new_df = apple_quote.filter(['Close'])
#Get the kast 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be vals between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Empty List
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to np array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[23]:


apple_quote2 = web.DataReader(tickerSymbol, data_source = 'yahoo', start = '2020-05-04', end = '2020-05-05')
print(apple_quote2)


# In[ ]:




