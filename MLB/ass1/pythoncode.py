# Importing the necessary libraries
import numpy as np
import pandas as pd

# Import the metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the dataset 
data = pd.read_csv('data_insurance.csv')
# First let us print the dimensions of the dataset 
print(data.shape)

# Print the head of the dataset to see what kind of data is available
print(data.head())

# Printing the info about the rows and columns of the dataset
print(data.info())

# To convert the region column, we will use one-hot encoding
df = pd.get_dummies(data)
print(df.head())   
# Now let us split the dataset into input features and output varaible
X = df.drop('charges', axis=1)
y = df['charges']

## Splitting the dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create an instance of the model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)
# Use the model on the training and testing data
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)


# Calculate the metrics
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

# Print the training and testing metrics
print("Training MSE: {:.4f}".format(train_mse))
print("Testing MSE: {:.4f}".format(test_mse))

print("\nTraining R^2: {:.4f}".format(train_r2))
print("Testing R^2: {:.4f}".format(test_r2))
