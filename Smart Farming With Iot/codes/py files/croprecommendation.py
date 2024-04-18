import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier,StackingClassifier, VotingClassifier, AdaBoostClassifier

from google.colab import files

upload = files.upload()

df = pd.read_csv('Crop_recommendation.csv')

x = df.drop(df.columns[0], axis=1)
x = x.round(1)
x.sample(5)

print('Number of instances = ',df.shape[0])
print('Number of attributes = ',df.shape[1],"\n")
# To get then number of missing values
print(df.isna().sum())

df.info()

df.describe()

numeric_columns = df.select_dtypes(include=[np.number]).columns
sns.boxplot(data=df[numeric_columns])
plt.show()

sns.relplot(x='rainfall',y='temperature',palette='rainbow',data=df,kind='scatter',hue='label')
plt.show()

crops = df['label'].unique()
print(len(crops))
print(crops)
print(pd.value_counts(df['label']))

df2=[]
for i in crops:
    df2.append(df[df['label'] == i])
df2[-1].head()

select = ['N', 'P','K','temperature', 'humidity', 'ph','rainfall']

for i in select:
  sns.catplot(data=df, x='label', y=i, kind='box', aspect=21/9, hue='label')
  plt.title(i.upper(), size=20)
  plt.xticks(rotation=45, ha='right')
  plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include=np.number)

# Plot heatmap
sns.heatmap(numeric_df.corr(), cmap='rainbow', annot=True)
plt.show()

def detect_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3-q1
    lower_limit = q1 - (1.5*IQR)
    upper_limit = q3 + (1.5*IQR)
    print("---\tOutlier Detection Information\t---\n")
    print(f"Q1 (25th percentile): {q1}")
    print(f"Q3 (75th percentile): {q3}")
    print(f"IQR (Interquartile Range): {IQR}")
    print(f"Lower limit for outliers: {lower_limit}")
    print(f"Upper limit for outliers: {upper_limit}")
    print(f"Minimum value: {x.min()}")
    print(f"Maximum value: {x.max()}")
    for i in [x.min(),x.max()]:
        if i == x.min():
            if lower_limit > x.min():
                print("Lower limit failed - Need to remove minimum value")
            elif lower_limit < x.min():
                print("Lower limit passed - No need to remove outlier")
        elif i == x.max():
            if upper_limit > x.max():
                print("Upper limit passed - No need to remove outlier")
            elif upper_limit < x.max():
                print("Upper limit failed - Need to remove maximum value")
detect_outlier(df['K'][df['label']=='grapes'])

for i in df['label'].unique():
    detect_outlier(df['K'][df['label']==i])
    print('---------------------------------------------\n')

X = df.drop(['label'], axis=1)
Y = df['label']

encode = preprocessing.LabelEncoder()
Y = encode.fit_transform(Y)

print("Label length Y: ",len(Y))
X.head()

x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y) #,test_size=0.3, random_state=42)

print("Length of x_train:", len(x_train))
print("Length of y_train:", len(y_train))
print("Length of x_test:", len(x_test))
print("Length of y_test:", len(y_test))

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

algorithms = {
    'decision tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'decisiontreeclassifier__criterion': ['gini', 'entropy'],
            'decisiontreeclassifier__splitter': ['best', 'random']
        }
    },
    'svm': {
        'model': SVC(),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'k_classifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'kneighborsclassifier__n_neighbors': [5, 10, 20, 25],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    },'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    },   'linear_regression': {
        'model': LinearRegression(),
        'params': {}
    },'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    }}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

score = []
details = []
best_param = {}

for mdl, par in algorithms.items():
    pipe = make_pipeline(preprocessing.StandardScaler(), par['model'])
    res = model_selection.GridSearchCV(pipe, par['params'], cv=5)

    start_time = time.time()  # Start time for training
    res.fit(x_train, y_train)
    end_time = time.time()    # End time for training

    # Predictions on test set for efficiency calculation
    start_pred_time = time.time()  # Start time for prediction
    y_pred = res.predict(x_test).astype(int)
    end_pred_time = time.time()    # End time for prediction

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    train_time = end_time - start_time
    test_time = end_pred_time - start_pred_time


    details_df = pd.DataFrame(res.cv_results_)
    details_df['Model name'] = mdl
    details.append(details_df)


    score.append({
        'Model name': mdl,
        'Best score': res.best_score_,
        'Best param': res.best_params_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 score': f1,
        'Training time (s)': train_time,
        'Testing time (s)': test_time
    })

    best_param[mdl]=res.best_estimator_

details_df = pd.concat(details, ignore_index=True)

score_df = pd.DataFrame(score)

details_df.head()

score_df.round(4)

score_df["Precision"]

models = score_df['Model name']
accuracy_scores = score_df['Accuracy'] * 100

accuracy_scores_filtered = [score for score in accuracy_scores if 90 <= score <= 100]
models_filtered = [models[i] for i, score in enumerate(accuracy_scores) if 90 <= score <= 100]

# Create a bar chart
plt.figure(figsize=[10, 5], dpi=100)
plt.xlabel("Models", fontweight="bold")
plt.ylabel("Accuracy", fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.title("Model Accuracy Comparison (90-100 Accuracy)")
sns.barplot(
    x=accuracy_scores_filtered,
    y=models_filtered,
    palette="rainbow",
)

# Set x-axis limits to 80 and 100
plt.xlim(90, 100)

plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create some sample data
models = score_df['Model name']
f1_scores =score_df['F1 score'] *100

# Filter for F1-score between 90 and 100
f1_scores_filtered = [score for score in f1_scores if 90 <= score <= 100]
models_filtered = [models[i] for i, score in enumerate(f1_scores) if 90 <= score <= 100]

# Create a bar chart
plt.figure(figsize=[10, 5], dpi=100)
plt.xlabel("Models", fontweight="bold")
plt.ylabel("F1", fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.title("Model F1-score Comparison (90-100)")
sns.barplot(
    x=f1_scores_filtered,
    y=models_filtered,
    palette="rainbow",
)

# Set x-axis limits to 90 and 100
plt.xlim(90, 100)

plt.show()

"""Naive Bayes has the highest accuracy and the hoghest F1 score so this algorithm has the best results use in thie project

"""

base_models = [
    ('decision_tree', algorithms['decision tree']['model']),
    ('svm', algorithms['svm']['model']),
    ('random_forest', algorithms['random_forest']['model']),
    ('k_classifier', algorithms['k_classifier']['model']),
    ('naive_bayes', algorithms['naive_bayes']['model']),
    ('logistic_regression', algorithms['logistic_regression']['model'])
]

from sklearn.preprocessing import StandardScaler

# Bagging
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging_model.fit(x_train, y_train)
bagging_pred = bagging_model.predict(x_test)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
bagging_f1 = f1_score(y_test, bagging_pred, average='weighted')
print("Bagging Accuracy:", bagging_accuracy)
print("Bagging F1 Score:", bagging_f1)

# Boosting
boosting_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
boosting_model.fit(x_train, y_train)
boosting_pred = boosting_model.predict(x_test)
boosting_accuracy = accuracy_score(y_test, boosting_pred)
boosting_f1 = f1_score(y_test, boosting_pred, average='weighted')
print("Boosting Accuracy:", boosting_accuracy)
print("Boosting F1 Score:", boosting_f1)

scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(x_train)

# Transform the test data
X_test_scaled = scaler.transform(x_test)

# Extract logistic regression model from algorithms dictionary
logistic_model = LogisticRegression(max_iter=2**63 - 1)

# Stacking
stacking_model = StackingClassifier(estimators=base_models, final_estimator=logistic_model)
stacking_model.fit(X_train_scaled, y_train)
stacking_pred = stacking_model.predict(X_test_scaled)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
stacking_f1 = f1_score(y_test, stacking_pred, average='weighted')

print("Stacking Accuracy:", stacking_accuracy)
print("Stacking F1 Score:", stacking_f1)

# Blending
X_train_blend, X_blend, y_train_blend, y_blend = model_selection.train_test_split(X_train_scaled, y_train, test_size=0.5, random_state=42)
blend_models = [clf.fit(X_train_blend, y_train_blend) for _, clf in base_models]
blend_pred = np.column_stack([model.predict(X_blend) for model in blend_models])
blend_meta_model = logistic_model.fit(blend_pred, y_blend)
blend_test_pred = blend_meta_model.predict(np.column_stack([model.predict(X_test_scaled) for model in blend_models]))
blend_accuracy = accuracy_score(y_test, blend_test_pred)
blend_f1 = f1_score(y_test, blend_test_pred, average='weighted')

print("Blending Accuracy:", blend_accuracy)
print("Blending F1 Score:", blend_f1)

max_original_accuracy = max(accuracy_scores_filtered)
better_accuracy = []
ensemble_accuracy = {"Bagging":bagging_accuracy, "Boosting": boosting_accuracy, "Stacking": stacking_accuracy,"Blending": blend_accuracy}
for item in ensemble_accuracy.keys():
  if ensemble_accuracy[item]*100 > max_original_accuracy:
    better_accuracy.append((item,ensemble_accuracy[item]*100))

max_original_f1 = max(f1_scores_filtered)
better_f1_accuracy = []
ensemble_f1 = {"Bagging":bagging_f1, "Boosting": boosting_f1, "Stacking": stacking_f1,"Blending": blend_f1}
for item in ensemble_f1.keys():
  if ensemble_f1[item]*100 > max_original_f1:
    better_f1_accuracy.append((item,ensemble_f1[item]*100))

import matplotlib.pyplot as plt
import seaborn as sns

# Original model
models = ['Original']

# Ensemble models with better accuracy
ensemble_models = [model for model, accuracy in better_accuracy]
models += ensemble_models

# Accuracy scores
accuracy_scores = [max_original_accuracy] + [accuracy for model, accuracy in better_accuracy]
if not better_f1_accuracy:
    print("The original had better accuracy ")
else:

  data = pd.DataFrame({'Models': models, 'Accuracy': accuracy_scores})

  # Set seaborn style and color palette
  sns.set(style="whitegrid")
  sns.set_palette("rainbow")

  # Create a pointplot
  sns.pointplot(data=data, x='Models', y='Accuracy')

  plt.xlabel("Models", fontweight="bold")
  plt.ylabel("Accuracy", fontweight="bold")
  plt.title("Maximum Accuracy Comparison")
  plt.xticks(rotation=45, ha="right")
  plt.grid(True)
  plt.show()

models_f1 = ['Original']

# Ensemble models with better F1 scores
ensemble_models_f1 = [model for model, f1_score in better_f1_accuracy]
models_f1 += ensemble_models_f1

# F1 scores
f1_scores = [max_original_f1] + [f1_score for model, f1_score in better_f1_accuracy]
if not better_f1_accuracy:
    print("The original had better f1 score than ensemble ones")
else:
  # Create a dataframe for Seaborn
  data_f1 = pd.DataFrame({'Models': models_f1, 'F1 Score': f1_scores})

  # Set seaborn style and color palette
  sns.set(style="whitegrid")
  sns.set_palette("rainbow")

  # Create a pointplot
  sns.pointplot(data=data_f1, x='Models', y='F1 Score')

  plt.xlabel("Models", fontweight="bold")
  plt.ylabel("F1 Score", fontweight="bold")
  plt.title("Maximum F1 Score Comparison")
  plt.xticks(rotation=45, ha="right")
  plt.grid(True)
  plt.show()
