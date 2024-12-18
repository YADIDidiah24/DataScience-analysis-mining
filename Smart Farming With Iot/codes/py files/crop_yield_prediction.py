
from google.colab import files

upload = files.upload()

import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df_yield = pd.read_csv('yield.csv')
df_yield.info()

df_yield

"""Looking at the columns in the csv, we can rename **Value** to **hg/ha_yield** to make it easier to recognise that this is our crops yields production value. In addition to removal of unnecessary coloumns like Area Code, Domain, Item Code, etc."""

# rename columns.
df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
df_yield.head()

# drop unwanted columns.
df_yield = df_yield.drop(['Year Code','Element Code','Element','Year Code','Area Code','Domain Code','Domain','Unit','Item Code'], axis=1)
df_yield.head()

df_yield.describe()

"""### Climate Data : Rainfall
The climatic factors include rainfall and temperature. They are abiotic components, including pesticides and soil, of the environmental factors that influence plant growth and development.


Rainfall has a dramatic effect on agriculture. For this project rain fall per year information was gathered from World Data Bank.
"""

df_rain = pd.read_csv('rainfall.csv')
df_rain.head()

df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})
df_rain = df_rain.rename(index=str, columns={"average_rain_fall_mm_per_year": 'Average Rainfall'})

df_rain.info()

"""convert average_rain_fall_mm_per_year from object to float"""

df_rain['Average Rainfall'] = pd.to_numeric(df_rain['Average Rainfall'],errors = 'coerce')

df_rain = df_rain.dropna()

df_rain.describe()

"""Merge Yield Dataframe with rain Dataframe by year and area columns"""

yield_df = pd.merge(df_yield, df_rain, on=['Year','Area'])

yield_df.head()

yield_df.describe()

"""### Pesticides Data:
Pesticides used for each item and country was also collected from FAO database.  
"""

df_pes = pd.read_csv('pesticides.csv')
df_pes.head()

df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
df_pes = df_pes.drop(['Element','Domain','Unit','Item'], axis=1)
df_pes.head()

df_pes.describe()

df_pes.info()

yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])
yield_df.shape

yield_df.head()

plt.figure(figsize=(10, 6))
plt.scatter(yield_df['Average Rainfall'], yield_df['hg/ha_yield'], c='blue', alpha=0.5)
plt.title('Relationship between Rainfall and Yield Production')
plt.xlabel('Average Rainfall')
plt.ylabel('Yield Production')
plt.grid(True)
plt.show()

df_temp=  pd.read_csv('temp.csv')
df_temp.describe()

df_temp = df_temp.rename(index=str, columns={"year": "Year", "country":'Area'})
df_temp.head()

df_temp = df_temp.dropna()

yield_df = pd.merge(yield_df,df_temp, on=['Area','Year'])
yield_df.head()

yield_df.shape

yield_df.isnull().sum()

"""**yield_df** is the final obtained dataframe;"""

yield_df.groupby('Item').count()

yield_df.describe()

"""It can be noticed the high variance in the values for each columns, later on I'll account for that will scaling."""

yield_df['Area'].nunique()

yield_df = yield_df.rename(index=str, columns={"Value": "hg/ha_yield"})

yield_df.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)

yield_df.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

correlation_data = yield_df.select_dtypes(include=[np.number]).corr()

# Creating a mask
mask = np.zeros_like(correlation_data, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Plotting
f, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(correlation_data, mask=mask, vmax=.3, center=0,
            square=True, linewidths=.5,cmap="rainbow", cbar_kws={"shrink": .5});
''

yield_df.sample(5)

from sklearn.preprocessing import OneHotEncoder

encoded_yield_df = pd.get_dummies(yield_df, columns=['Area',"Item"], prefix = ['Country',"Item"])
features=encoded_yield_df.loc[:, encoded_yield_df.columns != 'hg/ha_yield']
label=yield_df['hg/ha_yield']
features.head()

features = features.drop(['Year'], axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)

encoded_yield_df = encoded_yield_df.drop(['Year'], axis=1)

test_df=pd.DataFrame(test_data,columns=encoded_yield_df.loc[:, encoded_yield_df.columns != 'hg/ha_yield'].columns)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

clf=DecisionTreeRegressor()
model=clf.fit(train_data,train_labels)

test_df["yield_predicted"]= model.predict(test_data)
test_df["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.scatterplot(x=test_df["yield_actual"], y=test_df["yield_predicted"],cmap="rainbow")
sns.regplot(x=test_df["yield_actual"], y=test_df["yield_predicted"], scatter=False, color="red")
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Actual vs Predicted with Line of Best Fit")
plt.legend(['Data Points','Line of Best Fit'])
plt.show()

r2 = r2_score(test_df["yield_actual"], test_df["yield_predicted"])
print("R-squared:", r2)

yield_df.sample(5)

crop_yield_mean = yield_df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False)


top_5_crops = crop_yield_mean.head(10)

# Step 3: Filter the original DataFrame to include only the data for the top 5 crops
top_5_df = yield_df[yield_df['Item'].isin(top_5_crops.index)]

# Step 4: Create a bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Item', y='hg/ha_yield', palette="rainbow", data=top_5_df)
plt.title('Top 10 Crops with Highest Mean Yield')
plt.xlabel('Crop')
plt.ylabel('Mean Yield (hg/ha)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

yield_df.describe().round(2)

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np

# Base Models
base_models = [
    ('ridge', RidgeCV()),
    ('dt', DecisionTreeRegressor()),
    ('rf', RandomForestRegressor()),
    ('ada', AdaBoostRegressor()),
    ('gb', GradientBoostingRegressor())
]

# Stacking
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=RidgeCV())
ensemble_r2 = {}
# Blending
def blend_models(base_models):
    models = []
    for name, model in base_models:
        models.append((name, model))
    return VotingRegressor(models)

# Bagging (Random Forest)
bagging_model = RandomForestRegressor()

# Boosting (AdaBoost)
boosting_model = AdaBoostRegressor()

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Fit and evaluate each model
models = {
    'Stacking': stacking_regressor,
    'Blending': blend_models(base_models),
    'Bagging': bagging_model,
    'Boosting': boosting_model
}

for name, model in models.items():
    if name == 'Blending':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    ensemble_r2[name] = r2
    print(f"{name} R-squared: {r2}")

# Compare ensemble models with the original model
original_model = DecisionTreeRegressor()
original_model.fit(X_train, y_train)
y_pred_original = original_model.predict(X_val)
r2_original = r2_score(y_val, y_pred_original)
print(f"Original Model R-squared: {r2_original}")

max_ensemble_accuracy = max(ensemble_r2.values())
max_ensemble_model = [model for model, accuracy in ensemble_r2.items() if accuracy == max_ensemble_accuracy][0]
print("Model with maximum accuracy from ensemble models:", max_ensemble_model)

# Check if the best model object exists in the list of ensemble models
if max_ensemble_model in models:
    # Retrieve the model object corresponding to the name max_ensemble_model
    best_model_object = models[max_ensemble_model]

    # Use the model with maximum accuracy to make predictions on the test set
    best_model_object.fit(X_train, y_train)
    y_pred_max_ensemble = best_model_object.predict(X_val)

    # Plot actual vs. predicted values using the model with maximum accuracy
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_val, y=y_pred_max_ensemble, cmap="rainbow")
    sns.regplot(x=y_val, y=y_pred_max_ensemble, scatter=False, color="red")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f"Actual vs Predicted with Line of Best Fit (Best Model: {max_ensemble_model})")
    plt.legend(['Data Points','Line of Best Fit'])
    plt.show()
else:
    print("The best model is the original model.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data_r2 = pd.DataFrame({'Models': ['Original', max_ensemble_model], 'R-squared': [r2_original, max_ensemble_accuracy]})

# Set seaborn style and color palette
sns.set(style="whitegrid")
sns.set_palette("rainbow")

# Create a pointplot
sns.pointplot(data=data_r2, x='Models', y='R-squared')


plt.xlabel("Models", fontweight="bold")
plt.ylabel("R-squared", fontweight="bold")
plt.title("Maximum R-squared Comparison")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.show()

