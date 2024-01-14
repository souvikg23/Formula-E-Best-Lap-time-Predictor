#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib as plt
import pandas as pd
import numpy as np


# In[27]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install scipy')


# In[28]:


FE = pd.read_csv ('path//to//the//dataset',header = 0)


# In[29]:


print(FE.columns.tolist())


# In[30]:


FE.describe()


# In[31]:


print(FE)


# In[32]:


#check if there's any duplicate record
duplicate_check = FE.duplicated().sum()
print(duplicate_check)


# In[33]:


#check for null values
null_check = FE.isnull().sum()
print(null_check)


# In[34]:


df = FE[FE['Driver_Age'].isnull()]
print(df)


# In[35]:


#dropping the rows where a driver didn't finish
FE = FE[(FE['Race_Time'] != 'DNF') & (FE['Race_Time'] != '0')]

# Reset the index after dropping rows
FE.reset_index(drop=True, inplace=True)


print(FE)


# In[36]:





# Replacing missing 'Driver_Age' with values from other records of the same driver
FE['Driver_Age'] = FE.groupby('Driver')['Driver_Age'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# Reset the index after modifications
FE.reset_index(drop=True, inplace=True)


print(FE)


# In[37]:


df = FE[FE['Race_Time']=='DNS']
print(df)


# In[38]:


#dropping the rows where a driver didn't start
FE = FE[(FE['Race_Time'] != 'DNS')]

# Reset the index after dropping rows
FE.reset_index(drop=True, inplace=True)


print(FE)


# In[39]:


def convert_time(time_str):
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])
    total_seconds = minutes * 60 + seconds + milliseconds / 1000.0
    return minutes, seconds, milliseconds, total_seconds

# Convert 'Best_Lap_Time' and 'Race_Time' from string to second and minutes
FE[['Best_Lap_Min', 'Best_Lap_Sec', 'Best_Lap_Ms', 'Best_Lap_Total_Sec']] = FE['Best_Lap_Time'].apply(convert_time).apply(pd.Series)
FE[['Race_Min', 'Race_Sec', 'Race_Ms', 'Race_Total_Sec']] = FE['Race_Time'].apply(convert_time).apply(pd.Series)

# Display the updated DataFrame
print(FE)


# In[ ]:





# In[40]:


FE['Weather'] = FE['Weather'].str.extract('(\d+)').astype(float)


# In[41]:



FE['Track_Length'] = FE['Track_Length'].str.extract('([0-9.]+)').astype(float)


# In[16]:


print(FE)


# In[42]:


#driver numbers were in string, converting them to float
FE['Driver_Number'] = FE['Driver_Number'].str.extract('(\d+)').astype(float)


# In[84]:


print(FE)


# In[43]:


#now ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[44]:


# Convert categorical variables into numerical representations 
FE_enc = pd.get_dummies(FE, columns=['Team', 'Powertrain'])
print(FE_enc.columns.tolist())


# In[45]:


features = ['Turns', 'Weather', 'Track_Length', 'Driver_Age', 'Team_ABT Cupra Formula E Team', 'Team_Avalanche Andretti Formula E', 'Team_DS Penske', 'Team_Envision Racing', 'Team_Jaguar TCS Racing', 'Team_Mahindra Racing', 'Team_Maserati MSG Racing', 'Team_NEOM McLaren Formula E Team', 'Team_NIO 333 Racing', 'Team_Nissan Formula E Team', 'Team_Tag Heuer Porsche Formula E Team', 'Powertrain_DS E-TENSE FE23', 'Powertrain_Jaguar I-Type 6', 'Powertrain_Mahindra M9Electro', 'Powertrain_Maserati Tipo Folgore', 'Powertrain_NIO 333 ER9', 'Powertrain_Nissan e-4ORCE 04', 'Powertrain_Porsche 99X Electric Gen3', 'Starting_Position']
target = 'Best_Lap_Total_Sec'


# In[46]:


X = FE_enc[features]
y = FE_enc[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[47]:



model1 = LinearRegression()
model1.fit(X_train, y_train)
predictions = model1.predict(X_test)
mse1 = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse1}')


# In[ ]:





# In[114]:





# In[ ]:





# In[48]:


from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(random_state=42)
model2.fit(X_train, y_train)
predictions = model2.predict(X_test)
mse2 = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error 2nd : {mse2}')


# In[49]:


from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, predictions_rf)
print(f'Mean Squared Error (Random Forest): {mse_rf}')


# In[50]:


from sklearn.svm import SVR
model_svm = SVR()
model_svm.fit(X_train, y_train)
predictions_svm = model_svm.predict(X_test)
mse_svm = mean_squared_error(y_test, predictions_svm)
print(f'Mean Squared Error (Support Vector Machine): {mse_svm}')


# In[ ]:



team_list = ['ABT Cupra Formula E Team', 'Avalanche Andretti Formula E', 'DS Penske', 'Envision Racing', 'Jaguar TCS Racing', 'Mahindra Racing', 'Maserati MSG Racing', 'NEOM McLaren Formula E Team', 'NIO 333 Racing', 'Nissan Formula E Team', 'Tag Heuer Porsche Formula E Team']
powertrain_list = ['DS E-TENSE FE23', 'Jaguar I-Type 6', 'Mahindra M9Electro', 'Maserati Tipo Folgore', 'NIO 333 ER9', 'Nissan e-4ORCE 04', 'Porsche 99X Electric Gen3']

# Display the lists for user selection
print("Teams to choose from:")
for i, team in enumerate(team_list, start=1):
    print(f"{i}. {team}")

print("\nPowertrains to choose from:")
for i, powertrain in enumerate(powertrain_list, start=1):
    print(f"{i}. {powertrain}")

# User input
team_index = int(input("Enter the number corresponding to the chosen team: ")) - 1  # Subtracting 1 to convert to 0-based index
powertrain_index = int(input("Enter the number corresponding to the chosen powertrain: ")) - 1  # Subtracting 1 to convert to 0-based index
weather_input = float(input("Enter the weather in celcious: "))
track_length_input = float(input("Enter the track length in km: "))
driver_age_input = float(input("Enter the driver's age in year: "))

starting_position_input = int(input("Enter the starting position: "))
turns_input = int(input("Enter the number of turns on the track: "))

user_input_data = pd.DataFrame({
    'Turns': [turns_input],
    'Weather': [weather_input],
    'Track_Length': [track_length_input],
    'Driver_Age': [driver_age_input],
    
   
    **{f'Team_{team}': [0] for team in team_list},
    **{f'Powertrain_{powertrain}': [0] for powertrain in powertrain_list},
    'Starting_Position': [starting_position_input]
})

# Set the chosen team and powertrain to 1
user_input_data[f'Team_{team_list[team_index]}'] = 1
user_input_data[f'Powertrain_{powertrain_list[powertrain_index]}'] = 1

# Make predictions using the trained model
user_predictions = model_rf.predict(user_input_data)
print(f'Predicted BEST Lap Time: {user_predictions[0]} seconds')


# In[ ]:




