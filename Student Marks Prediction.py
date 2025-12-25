import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# LOAD THE DATA

load_data=pd.read_csv('exams.csv')

# DATA PREPROCESSING

new_data=load_data[['test preparation course','math score','reading score','writing score']]
# Fixing the nulls
new_data=new_data.dropna()

# Data encoding

new_data['test preparation course']=new_data['test preparation course'].map({'none':0,'completed':1})


# Data Splitting

X=new_data[['test preparation course','reading score','writing score']]  # for input training

Y=new_data[['math score']]

# TRAIN and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2, random_state=42
)

# training

Model=LinearRegression()
Model.fit(X_train,Y_train)


# predictions

Predictions=Model.predict(X_test).flatten()
Actual_Values=Y_test.values.flatten()
Comparision =pd.DataFrame({'Actual Values':Actual_Values,'Predictions':Predictions})


# Metrics


mae=mean_absolute_error(Y_test,Predictions)
mse=mean_squared_error(Y_test,Predictions)
rmse=np.sqrt(mse)
r2=r2_score(Y_test,Predictions)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")


# Graph is optional and but dont worry i show you how to create graphs
mlt.figure(figsize=(8,6))
mlt.scatter(Predictions,Actual_Values,color='blue',alpha=0.5,label="Datapoints")
mlt.show()