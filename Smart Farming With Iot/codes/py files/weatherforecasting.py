from google.colab import files

upload = files.upload()

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import missingno as msno
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import datetime as dt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet,Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import xgboost as xgb

df = pd.read_csv('DailyDelhiClimateTrain.csv')

df.info()

df.round(2).sample(5)

print('Number of instances = ',df.shape[0])
print('Number of attributes = ',df.shape[1],"\n")
# To get then number of missing values
print(df.isna().sum())

df['date'] = pd.to_datetime(df['date'])


fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='date', y='meantemp',palette="rainbow", data=df, ax=ax)


# Display the plot
plt.tight_layout()
plt.show()

df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')

df['dayOfWeek'] = df['date'].dt.dayofweek
df['year']=df['date'].dt.year
df['day']=df['date'].dt.day
df['month']=df['date'].dt.month

numeric_features = []
for column in df.columns:
    if df[column].dtype == 'float64':
        numeric_features.append(column)

print("Attributes with data type float64:")
print(numeric_features)

correlation = df.corr()
print(correlation['meantemp'].sort_values(ascending = False),'\n')

# Calculate correlation matrix
corr_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True,cmap='rainbow', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.show()

!pip install pmdarima

import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = df.drop(columns=['meantemp'])  # Features (excluding the target variable)
Y = df['meantemp']  # Target variable (mean temperature)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=100, shuffle=False)

plt.figure(figsize=(35, 5))
plt.grid()

# Plot train set
plt.plot(x_train.index, y_train, marker='v', label='Train')

# Plot test set
plt.plot(x_test.index, y_test, marker='o', label='Test')

plt.xticks(rotation=45)
plt.legend()
plt.show()

# Fit a simple auto_arima model
arima = pm.auto_arima(y_train, exogenous=x_train.drop(columns=['date']), d=2, error_action='ignore', trace=True,
                      suppress_warnings=True, maxiter=500, m=4, stationary=False, test='adf')

# Print out summary information on the fit
print(arima.summary())
print(arima.params)

# Generate predictions
pred = arima.predict(n_periods=100, exogenous=x_test.drop(columns=['date']), alpha=0.05)

plt.figure(figsize=(15,5))
plt.grid()

# Plot actual test values
plt.plot(x_test.index, y_test, marker='o', label='Test')

# Plot predicted values
plt.plot(x_test.index, pred, marker='v', label='Prediction')

plt.legend()
plt.xticks(rotation=90)
plt.show()

print("="*50)
print("="*50)

# Calculate evaluation metrics
mape = mean_absolute_percentage_error(y_test, pred)
print("Mean Absolute Percentage Error:", mape)
print("="*50)
mse = mean_squared_error(y_test, pred)
print("Mean Squared Error:", mse)
print("="*50)
print("Root Mean Squared Error:", mse ** 0.5)
print("="*50)
print("AIC:", arima.aic())

plt.figure(figsize=(15,5))
plt.grid()

# Plot actual test values
sns.lineplot(x=x_test.index, y=y_test,palette="rainbow", marker='o', label='Test')

# Plot predicted values
sns.lineplot(x=x_test.index, y=pred,palette="rainbow", marker='v', label='Prediction')

plt.legend()
plt.xticks(rotation=90)
plt.show()

print("="*50)
print("="*50)

# Calculate evaluation metrics
arima_mape = mean_absolute_percentage_error(y_test, pred)
print("Mean Absolute Percentage Error:", mape)
print("="*50)
arima_mse = mean_squared_error(y_test, pred)
print("Mean Squared Error:", mse)
print("="*50)
print("Root Mean Squared Error:", mse ** 0.5)
print("="*50)
arima_aic = arima.aic()
print("AIC:",arima_aic )
arima_rmse = arima_mse ** 0.5

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
# Convert datetime column to pandas datetime type if not already done
df['date'] = pd.to_datetime(df['date'])

# Extract year, month, day, and day of week features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayOfWeek'] = df['date'].dt.dayofweek

# Drop the original datetime column
df.drop(columns=['date'], inplace=True)

# Perform one-hot encoding or label encoding for categorical features if needed
# For example, if 'dayOfWeek' is treated as a categorical feature, you can use one-hot encoding
# df = pd.get_dummies(df, columns=['dayOfWeek'])

# Split the data into features (X) and target (Y)
X = df.drop(columns=['meantemp'])  # Features (excluding the target variable)
Y = df['meantemp']  # Target variable (mean temperature)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=100, shuffle=False)

# Now, you can proceed to fit your models

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('ridge', RidgeCV()),
    ('tree', DecisionTreeRegressor())
]

# Define meta-model
meta_model = LinearRegression()

# Define stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Fit stacking model
stacking_model.fit(x_train, y_train)

# Predict
stacking_pred = stacking_model.predict(x_test)

# Train base models
base_model1 = RandomForestRegressor(n_estimators=10, random_state=42)
base_model1.fit(x_train, y_train)

base_model2 = RidgeCV()
base_model2.fit(x_train, y_train)

base_model3 = DecisionTreeRegressor()
base_model3.fit(x_train, y_train)

# Combine predictions with weights
blending_pred = (0.3 * base_model1.predict(x_test)) + (0.4 * base_model2.predict(x_test)) + (0.3 * base_model3.predict(x_test))

from sklearn.ensemble import BaggingRegressor

# Define base model
base_model = RandomForestRegressor(n_estimators=10, random_state=42)

# Define bagging model
bagging_model = BaggingRegressor(base_model, n_estimators=5, random_state=42)

# Fit bagging model
bagging_model.fit(x_train, y_train)

# Predict
bagging_pred = bagging_model.predict(x_test)

import xgboost as xgb

# Define boosting model
boosting_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit boosting model
boosting_model.fit(x_train, y_train)

# Predict
boosting_pred = boosting_model.predict(x_test)


stacking_mape = mean_absolute_percentage_error(y_test, stacking_pred)
stacking_mse = mean_squared_error(y_test, stacking_pred)
stacking_rmse = stacking_mse ** 0.5
stacking_aic = arima.aic()

# Blending
blending_mape = mean_absolute_percentage_error(y_test, blending_pred)
blending_mse = mean_squared_error(y_test, blending_pred)
blending_rmse = blending_mse ** 0.5
blending_aic = arima.aic()

# Bagging
bagging_mape = mean_absolute_percentage_error(y_test, bagging_pred)
bagging_mse = mean_squared_error(y_test, bagging_pred)
bagging_rmse = bagging_mse ** 0.5
bagging_aic = arima.aic()

# Boosting
boosting_mape = mean_absolute_percentage_error(y_test, boosting_pred)
boosting_mse = mean_squared_error(y_test, boosting_pred)
boosting_rmse = boosting_mse ** 0.5
boosting_aic = arima.aic()

# Prepare tuples for each model and metric
metric_tuples = {
    "ARIMA": (arima_mape, arima_mse, arima_rmse, arima_aic),
    "Stacking": (stacking_mape, stacking_mse, stacking_rmse, stacking_aic),
    "Blending": (blending_mape, blending_mse, blending_rmse, blending_aic),
    "Bagging": (bagging_mape, bagging_mse, bagging_rmse, bagging_aic),
    "Boosting": (boosting_mape, boosting_mse, boosting_rmse, boosting_aic)
}

# Print metric values for each model
min_mape = min(metric_tuples.items(), key=lambda x: x[1][0])
min_mse = min(metric_tuples.items(), key=lambda x: x[1][1])
min_rmse = min(metric_tuples.items(), key=lambda x: x[1][2])
min_aic = min(metric_tuples.items(), key=lambda x: x[1][3])

print("Mean Absolute Percentage Error (MAPE) values:")
for model, metrics in metric_tuples.items():
    print(model, metrics[0])

print("="*50)
print("Mean Squared Error (MSE) values:")
for model, metrics in metric_tuples.items():
    print(model, metrics[1])

print("="*50)
print("Root Mean Squared Error (RMSE) values:")
for model, metrics in metric_tuples.items():
    print(model, metrics[2])

print("="*50)
print("AIC values:")
for model, metrics in metric_tuples.items():
    print(model, metrics[3])

# Print max values
print("="*50)
print("Best Mean Absolute Percentage Error (MAPE):", min_mape)
print("="*50)
print("Best Mean Squared Error (MSE):", min_mse)
print("="*50)
print("Max Root Mean Squared Error (RMSE):", min_rmse)
print("="*50)
print("Max AIC:", min_aic)

"""Metric	ARIMA	Stacking	Blending	Bagging	Boosting
Mean Absolute Percentage Error (MAPE)	0.056	0.093	0.118	0.093	0.081
Mean Squared Error (MSE)	2.277	8.513	9.244	9.113	6.453
Root Mean Squared Error (RMSE)	1.509	2.918	3.040	3.019	2.540
Akaike Information Criterion (AIC)	5469.426	5469.426	5469.426	5469.426	5469.426

"""

