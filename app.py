###### Libraries #######
# Base Libraries
import pandas as pd
import numpy as np

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########
data = pd.read_csv('Data.csv')

########### Loading Trained Model Files ########

model = joblib.load("Clickadd_xgb.pkl")
model_ohe = joblib.load("Clickadd_ohe.pkl")
model_sc = joblib.load("Clickadd_sc.pkl")
model_pca = joblib.load("Clicked_pca.pkl")
########## UI Code #############

# Ref: https://docs.streamlit.io/library/api-reference

# Title
st.header("Identifying weather the person clicked the add or not:")

# Image
with st.columns(3)[1]:
    st.image("https://dgbijzg00pxv8.cloudfront.net/ab1fe4b5-510e-4405-ad12-234afb1dfda9/000000-0000000000/39072274470874426121284330130872841955733143294728805461369592993420210902166/ITEM_PREVIEW1.png", width=400)

# Description
st.write("""Built a Predictive model in Machine Learning to Identify weather the User clicked the add or not.
         Sample Data taken as below shown.
""")

# column name spaces repalcing with ( _ )

data.columns = data.columns.str.replace(" ","_")
# deleting output column
del data['Clicked_on_Ad']
# timestamp is divided
from datetime import datetime

data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d-%m-%Y %H.%M")

data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Weekday'] = data['Timestamp'].dt.dayofweek
data['Hour'] = data['Timestamp'].dt.hour
data = data.drop(['Timestamp'], axis=1)

# Data Display
st.dataframe(data.head())
st.write("From the above data , Add clicked is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get weather the user clicked or not:")

col1,  col2,  col3 = st.columns(3) # value inside brace defines the number of splits
col4,  col5,  col6 = st.columns(3)
col7,  col8,  col9 = st.columns(3)
col10, col11, col12 = st.columns(3)


with col1:
    Daily_Time_Spent_on_Site = st.number_input("Enter Daily_Time_Spent_on_Site :")
    st.write(Daily_Time_Spent_on_Site)

with col2:
    st.write("If Age between 19 and 35 give Adults")
    st.write("If Age between 36 and 50 give Middle_Aged")
    st.write("If Age greater than 50 give Senior_Citizen")
    for i in range(len(data)):
          val = data.Age
          if val[i]>=19 and val[i]<=35:
               data.Age[i] = 'Adults'
          elif val[i]>35 and val[i]<=50:
               data.Age[i] = 'Middle_Aged'
          elif val[i]>50:
               data.Age[i] = 'Senior_Citizen'
    Age= st.selectbox("Enter Age:",data.Age.unique())
    st.write(Age)

with col3:
     Area_Income = st.number_input("enter Area_Income:")
     st.write(Area_Income)

with col4:
     Daily_Internet_Usage= st.number_input("enter Daily_Internet_Usage:")
     st.write(Daily_Internet_Usage)

with col5:
     Ad_Topic_Line= st.selectbox("enter Ad_Topic_Line :",data.Ad_Topic_Line.unique())
     st.write(Ad_Topic_Line)

with col6:
     City= st.selectbox("enter City:",data.City.unique())
     st.write(City)

with col7:
     Gender = st.selectbox("Enter Gender:",data.Gender.unique())
     st.write(Gender)

with col8:
     Country = st.selectbox("Enter Country:",data.Country.unique())
     st.write(Country)

with col9:
     Month= st.number_input("enter Month:")
     st.write(Month)

with col10:
     Day= st.number_input("enter Day:")
     st.write(Day)

with col11:
     Weekday= st.number_input("enter Weekday:")
     st.write(Weekday)

with col12:
     for i in range(len(data)):
          if data.Hour[i] >= 0  and data.Hour[i] <=5:
               data['Hour'][i] ='mid_night'
          elif data.Hour[i] >= 6 and data.Hour[i] <=  11:
               data['Hour'][i] = 'morning'
          elif data.Hour[i] >= 12 and data.Hour[i] <= 16:
               data['Hour'][i] = 'afternoon'
          elif data.Hour[i] >= 17 and data.Hour[i] <= 19:
               data['Hour'][i] = 'evening'
          else :
               data['Hour'][i] = 'night'
     Hour=st.selectbox("enter Hour:",data.Hour.unique())
     st.write(Hour)
###### Predictions #########

if st.button("Check here"):
    st.write("Data Given:")
    values = [Daily_Time_Spent_on_Site, Age,Area_Income,Daily_Internet_Usage,Ad_Topic_Line,City,Gender,Country,Month,Day,Weekday,Hour]
    record =  pd.DataFrame([values],
                           columns = ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income',
                                     'Daily_Internet_Usage', 'Ad_Topic_Line', 'City', 'Gender', 'Country',
                                     'Month', 'Day', 'Weekday', 'Hour'])
    for col in record.columns:
         if record[col].dtype=='object':
              record[col] = record[col].str.lower()
    st.dataframe(record)
    record.Age.replace({'adults':0,'middle_aged':1,'senior_citizen':2}, inplace = True)
    record.Gender.replace({'male':0,'female':1},inplace = True)
    record.Hour.replace({'morning':0,'afternoon':1,'evening':2,'night':3,'mid_night':4},inplace = True)
    ohedata = model_ohe.transform(record[['Ad_Topic_Line','City','Country']]).toarray()
    ohedata = pd.DataFrame(ohedata, columns = model_ohe.get_feature_names_out())
    record = pd.concat([record.iloc[:,0:],ohedata], axis = 1)
    record.drop(['Ad_Topic_Line','City','Country'], axis = 1,inplace =True)
    st.dataframe(record)
    record.iloc[:, [0,2,3,5,6,7,8]] = model_sc.transform(record.iloc[:, [0,2,3,5,6,7,8]])
    data_pca = model_pca.transform(record)
    data_pca = pd.DataFrame(data_pca[:,:43])


    clicked = model.predict(data_pca)[0]
    clicked = {1:'Clicked',0:'Not_Clicked'}[clicked]
    st.subheader("Clicked add or not : ")
    st.subheader(clicked)


