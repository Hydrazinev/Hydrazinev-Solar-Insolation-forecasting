# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:57:17 2023

@author: 91635
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.impute import KNNImputer

#Creating a dataset
start_date = datetime.datetime(2016, 1, 1, 0, 0)  # Starting date and time
end_date = datetime.datetime(2022, 12, 30, 23, 30)  # Ending date and time
frequency = datetime.timedelta(minutes=30)  # Time interval

date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)#giving range to generated data

df = pd.DataFrame({'date_time': date_range})#coverting the generated data into dataframe by assigning date_time variable

df['date_time'] = pd.to_datetime(df['date_time'])#converting data into date_time format and indexing it
df = df.set_index('date_time')

input_directory = r"/home/guest/Vaidik/vaidik"  # Use raw string (r) to preserve backslashes
file_name = "jodhpur_insolation.csv" 

file_path = os.path.join(input_directory, file_name)
file_path = os.path.abspath(file_path)#retriving the exsisting data from directory

df1 = pd.read_csv(file_path)
df1['date_time'] = pd.to_datetime(df1['date_time'])#reading the existing data and converting it in the datetimr format and indexing it
df1 = df1.set_index('date_time')

df = pd.merge(df,df1,on='date_time', how ='left') #merging the date_time from both generated data and existing data
df = df[(df.index.time >= pd.to_datetime('7:30').time()) &  #taking the times between 7am to 6:30 pm only
                  (df.index.time <= pd.to_datetime('18:30').time())]

c1 = df.isnull().sum()
df.to_csv('missingstamps_filled.csv') #cheking null values in the table and adding into the missingstamps_filled csv file

series = df.groupby(df.index.date)['ins'].count() #counting the number of blocks in generated data
print(series)

dt_list = []
for index , row in series.items(): # if there are less than 20 timestamp in a day it will be added in the dt_list
    if row<20:
        dt_list.append(index)
        print(index , row)
        del_date = index
        cond = df[df.index.date==del_date].index
        df.drop(cond , inplace = True)

df = df.reset_index(drop=False)
print("Dates count:", len(dt_list))#it will provide and give a number how many dates have less than 20 time stamps
count1 = df['ins'].isna().sum()


for i, row in df.iterrows():
    prod_list = []
    arg_list = []
    val = row['ins']
    if not i == 0:
        if pd.isnull(df.loc[i]['ins']):  #this is the condition if a row in a specific coloumn is null
            if not pd.isnull(df.loc[i-1]['ins']): #it checks for the previous value in location
                if not pd.isnull(df.loc[i+1]['ins']): #it checks for the next value in location  
                    df.loc[i, 'ins'] = round(((df.loc[i-1]['ins'] + df.loc[i+1]['ins']) / 2),3) #it averages both the location to fill the missing value 
                    print("Replaced Value:", df.loc[i, 'ins']) #missing value is assigned 
                    print("Timestamp:", df.index[i]) 
                else:
                    if not pd.isnull(df.loc[i+2]['ins']): #it checks 2 location next from the  missing value 
                        df.loc[i, 'ins'] =  round(((df.loc[i-1]['ins'] + df.loc[i+2]['ins']) / 2),3) #it averages both previous location and 2 location next for the missing value
                        print("Replaced Value:", df.loc[i, 'ins'])  #missing value is assigned
                        print("Timestamp:", df.index[i])
                    else:
                        # print("prev1 available, next two neighbours not available")
                        if not pd.isnull(df.loc[i-26]['ins']): # it checks for the previous day from the same time stamp
                            if not pd.isnull(df.loc[i+26]['ins']):# it checks for the next day from the same time stamp
                                df.loc[i, 'ins'] =  round(((df.loc[i-26]['ins'] + df.loc[i+26]['ins'])/2),3)#it averages both next day and previous days for the missing value
                                print("Replaced Value:", df.loc[i, 'ins']) #missing value is assigned
                                print("Timestamp:", df.index[i])
                            else:
                                #print("26th values not available")
                                if not pd.isnull(df.loc[i-27]['ins']):   #it checks for the one time stamp before for the prevoius day
                                    if not pd.isnull(df.loc[i-25]['ins']): #it checks for the one time stamp after for the prevoius day
                                        df.loc[i ,'ins'] =  round(((df.loc[i-27]['ins'] + df.loc[i-25]['ins'])/2),3) #it averages the above data
                                        print("Replaced Value:", df.loc[i, 'ins'])
                                        print("Timestamp:", df.index[i])
                                        print("Previous value:", df.loc[i-27]['ins'])
                                        print("Next value:", df.loc[i-25]['ins'])
                                    else:
                                        print("25th value not available")
                                else:
                                    print("27th value not available")
                        else:
                            #print("26th values not available")
                            if not pd.isnull(df.loc[i-27]['ins']):
                                if not pd.isnull(df.loc[i-25]['ins']):
                                    df.loc[i ,'ins'] =  round(((df.loc[i-27]['ins'] + df.loc[i-25]['ins'])/2),3)
                                    print("Replaced Value:", df.loc[i, 'ins'])
                                    print("Timestamp:", df.index[i])
                                    print("Previous value:", df.loc[i-27]['ins'])
                                    print("Next value:", df.loc[i-25]['ins'])
                                else:
                                    print("25th value not available")
                            else:
                                print("27th value not available")
            else:
                if not pd.isnull(df.loc[i-2]['ins']):
                    if not pd.isnull(df.loc[i+1]['ins']):
                        df.loc[i, 'ins'] =  round(((df.loc[i-2]['ins'] + df.loc[i+1]['ins']) / 2),3)
                        print("Replaced Value:", df.loc[i, 'ins'])
                        print("Timestamp:", df.index[i])
                    else:
                        if not pd.isnull(df.loc[i+2]['ins']):
                            df.loc[i, 'ins'] =  round(((df.loc[i-2]['ins'] + df.loc[i+2]['ins']) / 2),3)
                            print("Replaced Value:", df.loc[i, 'ins'])
                            print("Timestamp:", df.index[i])
                        else:
                            # print("prev1, prev2 not available")
                            if not pd.isnull(df.loc[i-26]['ins']):
                                if not pd.isnull(df.loc[i+26]['ins']):
                                    df.loc[i, 'ins'] =  round(((df.loc[i-26]['ins'] + df.loc[i+26]['ins'])/2),3)
                                    print("Replaced Value:", df.loc[i, 'ins'])
                                    print("Timestamp:", df.index[i])
                                else:
                                    #print("26th values not available")
                                    if not pd.isnull(df.loc[i-27]['ins']):
                                        if not pd.isnull(df.loc[i-25]['ins']):
                                            df.loc[i ,'ins'] = round(((df.loc[i-27]['ins'] + df.loc[i-25]['ins'])/2),3)
                                            print("Replaced Value:", df.loc[i, 'ins'])
                                            print("Timestamp:", df.index[i])
                                            print("Previous value:", df.loc[i-27]['ins'])
                                            print("Next value:", df.loc[i-25]['ins'])
                                        else:
                                            print("25th value not available")
                                    else:
                                        print("27th value not available")
                                    
                            else:
                                #print("26th values not available")
                                if not pd.isnull(df.loc[i-27]['ins']):
                                    if not pd.isnull(df.loc[i-25]['ins']):
                                        df.loc[i ,'ins'] =  round(((df.loc[i-27]['ins'] + df.loc[i-25]['ins'])/2),3)
                                        print("Replaced Value:", df.loc[i, 'ins'])
                                        print("Timestamp:", df.index[i])
                                        print("Previous value:", df.loc[i-27]['ins'])
                                        print("Next value:", df.loc[i-25]['ins'])
                                    else:
                                        print("25th value not available")
                                else:
                                    print("27th value not available")
    else:
        pass


c3 = df.isnull().sum()
df_filled = df.copy()
columns_to_impute = ['ins']
imputer = KNNImputer(n_neighbors=5)
df_filled[columns_to_impute] = imputer.fit_transform(df_filled[columns_to_impute])
   
df2 = df['ins'].interpolate()
df['ins']=df2

print("Check for any no data values---")

print(df['ins'].isnull().sum())      

df.to_csv("jodhpur_solar_insolation_2016_2022_ffinall.csv")