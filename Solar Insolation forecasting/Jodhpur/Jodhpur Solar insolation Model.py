# -*- coding: utf-8 -*-
#Importing the libararies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, SGD,RMSprop
from tensorflow.keras.models import load_model
from tensorflow import keras


import warnings
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
warnings.filterwarnings('ignore')
# import psycopg2

import datetime


# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


base_dir = r"/home/guest/Vaidik/vaidik"
dt = datetime.datetime.now()
dt_str = datetime.datetime.strftime(dt, "%d%m%Y_%H%M")
date = datetime.datetime.strftime(dt, "%d%m%Y")

filename = "insol_model_" + dt_str +".h5"
model_path = os.path.join(base_dir,date, filename)

print("Model Path ::::: ", model_path)

input_file_path = os.path.join(base_dir,"jodhpur_solar_insolation_2016_2022_main_up.csv")
df = pd.read_csv(input_file_path)

c = df.isnull().sum()
# Convert 'date_time' column to datetime format
df['date_time'] = pd.to_datetime(df['date_time'])

#Filter the data for the desired time range (7:00 to 19:30)
df = df[(df['date_time'].dt.time >= pd.to_datetime('7:30').time()) & (df['date_time'].dt.time <= pd.to_datetime('18:30').time())]
c1 = df.isnull().sum()
### BELOW DATA IS FOR TRAINING THE MODEL
train =df[df['date_time'].dt.year<2021]
train = train[train['date_time'].dt.month >=1]
train = train[train['date_time'].dt.month <=12]
test = df[df['date_time'].dt.year>=2021]

# split data into train before 2021 and test after 2021 

### BELOW DATA IS FOR INITIALIZING THE MODEL
# train =df[df['date_time'].dt.year<2018]
# test = df[df['date_time'].dt.year==2018]
# test = test[test['date_time'].dt.month > 4]
# test = test[test['date_time'].dt.month < 10]
train= train.set_index('date_time') # setting the index to date_time
test = test.set_index('date_time') # setting the index to date_time

# # simple scaling using max_value on train
# max_value = np.max(df['ins'])
# c1 = 1/max_value # constant of scaling
# data = train*c1 # scaled data 
# df_values = train['ins'].values # nd array of scaled values  
# print(df_values[12:20])
# df_train_scaled = data['ins'].values
# print(df_train_scaled[12:20])

# data = test*c1
# df_test_unscaled = test['ins'].values
# print(df_test_unscaled[12:20])
# df_test_scaled = data['ins'].values
# print(df_test_scaled[12:20])
# normalizing the data using minmaxscaler

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train['ins_filled'].values.reshape(-1, 1))
test_scaled = scaler.fit_transform(test['ins_filled'].values.reshape(-1, 1))

# splitting data into features and target variable

def split_sequence(sequence , n_steps_in ,  n_steps_out):
  X,y = [] , []
  for i in range(len(sequence)):
    end_ix = i + n_steps_in
    if end_ix > len(sequence)-1:
      break
    end_ox = end_ix + n_steps_out
    if end_ox > len(sequence)-1:
      break
    seq_x , seq_y = sequence[i:end_ix], sequence[end_ix:end_ox]
    X.append(seq_x)
    y.append(seq_y)

  return np.array(X) , np.array(y)

# setting the parameters of model
i = 24 # number of inputs per day 
n_steps_in = i*7 # 7 days of input
n_steps_out = i*2 # 2 days of output
n_features=1 # reason : univariate data
e=50


X_train, y_train = split_sequence(train_scaled, n_steps_in, n_steps_out) # training data split
X_test, y_test = split_sequence(test_scaled, n_steps_in, n_steps_out) # testing data split 

#reshaping the model to 3 dimensions 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
print("Shape of input : ", X_train.shape)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
print("Shape of input : ", X_test.shape)

# Define the paths for model checkpoint and early stopping
checkpoint_path = os.path.join(base_dir,date, model_path)
early_stopping_path = os.path.join(base_dir, date, 'early_stopping_checkpoint.h5')

# Model Checkpoint Callback: Save the model weights during training
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_r2_score',  # Choose a metric to monitor
    save_best_only=True,  # Save only the best model
    mode='max',  # 'min' for loss, 'max' for accuracy, 'auto' will be inferred
    verbose=1
)

# Early Stopping Callback: Stop training if the validation loss stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,
    verbose=1)

# Learning Rate Scheduler Callback: Adjust the learning rate during training
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,  # Factor by which the learning rate will be reduced (new_lr = lr * factor)
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-6,  # Minimum learning rate
    verbose=1
)

# Combine all callbacks in a list
callbacks = [checkpoint_callback, reduce_lr_callback, early_stopping_callback]

model = Sequential()
model.add(LSTM(128,input_shape=(n_steps_in, n_features), return_sequences=True, use_bias=True, activation='tanh', recurrent_activation='sigmoid'))
model.add(Dropout(0.2))  # Add dropout layer with dropout rate of 0.2
model.add(LSTM(128, return_sequences=False, use_bias=True, activation='tanh', recurrent_activation='sigmoid'))
model.add(Dropout(0.2))  # Add dropout layer with dropout rate of 0.2

model.add(Dense(n_steps_out, activation='relu'))


rmsprop = RMSprop(learning_rate=0.01)
adam = Adam(learning_rate=0.001)
sgd = SGD(learning_rate = 0.008, momentum=0.9)

def r2_score(y_true, y_pred): 
  total_variance = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
  residual_variance = tf.reduce_sum(tf.square(y_true - y_pred)) 
  r2 = 1 - (residual_variance / total_variance) 
  return r2 

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae', 'mse', r2_score])
#model.compile(optimizer = 'sgd', loss = keras.losses.Huber(), metrics = ['mae', r2_score])
history1 = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), callbacks=callbacks)

# model.compile(optimizer = sgd, loss = keras.losses.Huber(), metrics = ['mae', 'accuracy'])
# history2 = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# # model.load_weights('./model_checkpoint.h5')
# sgd2 = SGD(learning_rate = 0.01, momentum = 0.9)
# model.compile(optimizer=sgd2, loss=keras.losses.Huber(), metrics=['mae',r2_score])
# history2  = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks)


# model.summary()
#model.save(model_path, save_format='h5')


########################
###
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_pred = model.predict(X_test)

# Reshape the predictions and ground truth for comparison
y_pred = y_pred.reshape(-1, n_steps_out)
y_true = y_test.reshape(-1, n_steps_out)

# Inverse transform the scaled predictions and ground truth
y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y_true)

# Calculate Mean Squared Error (MSE) on the original scale
mse = mean_squared_error(y_true_inv, y_pred_inv)

r2 = r2_score(y_true_inv, y_pred_inv)
print(f"MSE :{mse}")
print(f'R_Squared : {r2}')
#print(f"MAE : {mae}")


# function to append 6 values ie every 3 hours 
x=840
true, pred = [], []
for i in range(10):
  for j in range(6):
    true.append(y_true_inv[x + 6*i][j])
    pred.append(y_pred_inv[x + 6*i][j])

plt.plot(true, label='True')
plt.plot(pred, label='Predicted')
plt.legend()
plt.title('Comparison of True vs Predicted Values')
plt.show()
plt.savefig(f'{mse}_true_pred.png')
# X_input_1 = np.array(X_train[0][-n_steps_x:], dtype=float)
# X_input = X_input_1.reshape((1, n_steps_x, n_features))

# y_pred_scaled = model.predict(X_input, verbose=0)
# yhat_unscaled = (y_pred_scaled * max_value ) # scaling up the predicted values

# exp = X_test[0][:n_steps_y] * max_value

# for i in range(n_steps_y):
#     print(yhat_unscaled[0][i], "\t\t", exp.T[0][i]) 
#     # transposing the expected array (exp) because it ha sshape 52,1, yhat_unscaled has shape 1,52

# v= []
# for i in range(n_steps_y):
#     v.append(i)
    
# ##################
    
plt.figure(figsize=(20,3))

plt.subplot(1,2,1)
ax = plt.gca().twinx()
plt.plot(history1.history['val_loss'])
plt.ylim([0.00,1.00])
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')

plt.plot(history1.history['val_r2_score'])
plt.ylabel('Epochs')
plt.xlabel('Validation Accuracy')
plt.tight_layout()
plt.savefig(f"/home/guest/Vaidik/vaidik/{dt_str}_{mse}_1.png")
plt.show()

