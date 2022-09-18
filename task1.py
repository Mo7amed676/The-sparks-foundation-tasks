"""
@author: Eng-M. Mahmoud
"""
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn import metrics
#read data
data=pd.read_csv("students.csv")
print(data.head())
#split data input and output X,y
X = data.iloc[:, :-1].values  ##( : )all rows ( :-1) from start to before last element
y = data.iloc[:, 1].values
#split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 
print(f"X_train: {X_train}","\n",f"X_test: {X_test}")

# # Plotting the distribution of scores
# data.plot(x='Hours', y='Scores', style='o')
# plt.title('Hours vs Percentage')
# plt.xlabel('Hours Studied')
# plt.ylabel('Percentage Score')
# plt.show()
# print(X_test) # Testing data - In Hours

# # Model Training
model = LinearRegression()  
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) # Predicting the scores

# # Plotting the regression line
# line = model.coef_*X+model.intercept_
# #Plotting for the test data
# plt.scatter(X, y)
# plt.plot(X, line);
# plt.show()

# Comparing Actual vs Predicted
df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
    }
)  
print(df)

# You can also test with your own data
hours = np.array([9.25]).reshape(1,1) ##matrix([9.25])
own_pred = model.predict(hours)

print(f"No of Hours = {hours}")
print(f"Predicted Score = {own_pred[0]}")
# print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
