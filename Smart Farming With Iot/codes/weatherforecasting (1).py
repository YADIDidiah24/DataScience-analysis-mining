
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
mape = mean_absolute_percentage_error(y_test, pred)
print("Mean Absolute Percentage Error:", mape)
print("="*50)
mse = mean_squared_error(y_test, pred)
print("Mean Squared Error:", mse)
print("="*50)
print("Root Mean Squared Error:", mse ** 0.5)
print("="*50)
print("AIC:", arima.aic())

