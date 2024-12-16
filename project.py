import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
def mean_absolute_error(prediction,target):
    return np.mean(np.abs(prediction-target))
def mean_squared_error(prediction,target):
    return np.mean(np.square(prediction-target))
def addColumn(years):
    values = np.random.rand()
    #In major software companies, on average, one project is assigned every quarter. Thus, the
    # synthetic data is 4+some random value between 0 and 1 multiplied to the number of years of experience
    # using np.floor to get an integer value.
    return np.floor((4+values)*years)
df = pd.read_csv('employee.csv')
print(df)
matplotlib.rcParams['figure.figsize'] = [10,6]

plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
print(df.describe())
#This function splits the data randomly which explains why we get a different error on every run
input_training,input_testing,target_training,target_testing = train_test_split(df['YearsExperience'],df['Salary'],test_size=0.2)
inputs = pd.DataFrame(input_training)
targets = pd.DataFrame(target_training)
input_tests = pd.DataFrame(input_testing)

model = LinearRegression()
model.fit(inputs,targets)
coefficients = model.coef_
intercept = model.intercept_
print("The coefficient is :",coefficients.flatten()[0],"\n The intercept is: ",intercept[0],'for the Single Regression model')
predictions = model.predict(input_tests)
print('Mean Absolute Error value: ',mean_absolute_error(predictions.flatten(),target_testing))
print('Mean Squared Error value: ',mean_squared_error(predictions.flatten(),target_testing))
print('Both for the Single Regression model')

predictions_total = model.predict(df[['YearsExperience']])


plt.scatter(df['YearsExperience'], df['Salary'],color='blue')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Comparison between predictions of the Single Regression Model and the actual values',fontsize=10)
plt.plot(df['YearsExperience'], predictions_total,color='red')
plt.legend(['Actual Salary','Predicted Salary'])
plt.show()

# Bonus Task

df['NumberOfProjects'] = addColumn(df['YearsExperience'])
input_train,input_test,target_train,target_test  = train_test_split(df[['NumberOfProjects','YearsExperience']],df['Salary'],test_size=0.2)
model2 = LinearRegression()
model2.fit(input_train,target_train)
prediction2 = model2.predict(input_test)
prediction_total = model2.predict(df[['NumberOfProjects','YearsExperience']])
print('The MAE: ',mean_absolute_error(prediction2.flatten(),target_test),'The MSE: ',mean_squared_error(prediction2.flatten(),target_test),'both for the multiple regression model')
plt.clf()
ax = plt.axes(projection='3d')
ax.scatter(df['YearsExperience'],df['NumberOfProjects'],df['Salary'],color='blue')
ax.plot(df['YearsExperience'],df['NumberOfProjects'],prediction_total.flatten(),color='red')
plt.legend(['Actual Salary','Predicted Salary'])
plt.title('Comparison between predictions of the Multiple Regression Model and the actual values',fontsize=10)
plt.xlabel('Years Experience')
plt.ylabel('NumberOfProjects')
ax.set_zlabel('Salary')
plt.show()
plt.clf()
plt.bar(x = ['Single Regression Model','Multiple Regression Model'],height = [mean_absolute_error(predictions.flatten(),target_testing),mean_absolute_error(prediction2.flatten(),target_test)])
plt.title('Comparing the mean absolute error of the Single Regression model and the Multiple Regression model')
plt.show()
plt.clf()
plt.bar(x = ['Single Regression Model','Multiple Regression Model'],height = [mean_squared_error(predictions.flatten(),target_testing),mean_squared_error(prediction2.flatten(),target_test)])
plt.title('Comparing the mean squared error of the Single Regression model and the Multiple Regression model')
plt.show()