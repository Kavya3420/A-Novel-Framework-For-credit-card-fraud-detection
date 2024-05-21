#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
import seaborn as sns


# In[2]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')


# In[3]:


# first 5 rows of the dataset
credit_card_data.head()


# In[4]:


def hist(x):
    plt.hist(credit_card_data[x], bins=25)
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming credit_card_data is your DataFrame
numeric_columns = credit_card_data.select_dtypes(include=["float64", "int64"]).columns

for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(credit_card_data[col], bins=30, kde=False, color='skyblue')  # Use a color name or code

    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f'Histogram of {col}', fontsize=14)
    plt.show()


# In[6]:


credit_card_data.tail()


# In[7]:


# dataset informations
credit_card_data.info()


# In[8]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[9]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[149]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[150]:


print(legit.shape)
print(fraud.shape)


# In[151]:


# statistical measures of the data
legit.Amount.describe()
     


# In[152]:


fraud.Amount.describe()


# In[153]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[154]:


legit_sample = legit.sample(n=492)


# In[155]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)
     

new_dataset.head()


# In[156]:


new_dataset.tail()


# In[157]:


new_dataset['Class'].value_counts()


# In[158]:


new_dataset.groupby('Class').mean()


# In[159]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
     


# In[160]:


print(X)


# In[161]:


print(Y)


# In[162]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
     

print(X.shape, X_train.shape, X_test.shape)


# In[163]:


model = LogisticRegression()
     

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
     


# In[164]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
training_data_precision = precision_score(X_train_prediction, Y_train)
training_data_recall = recall_score(X_train_prediction, Y_train)
training_data_f1_score = f1_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
print('precision on Training data: ',training_data_precision)
print('Recall on Training data: ',training_data_recall)
print('F1 score on Training data: ',training_data_f1_score)


# In[165]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
test_data_precision = precision_score(X_test_prediction, Y_test)
test_data_recall = recall_score(X_test_prediction, Y_test)
test_data_f1_score = f1_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
print('precision score on Test Data:',test_data_precision)
print('Recall score on Test Data:',test_data_recall)
print('F1 score on Test Data:',test_data_f1_score)


# In[166]:


# Initialize the RandomForestClassifier
model = RandomForestClassifier()

# Train the random forest classifier with Training Data
model.fit(X_train, Y_train)


# In[167]:


# Check the dimensions of X_train and Y_train
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# In[168]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
training_data_precision = precision_score(X_train_prediction, Y_train)
training_data_recall = recall_score(X_train_prediction, Y_train)
training_data_f1_score = f1_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
print('precision on Training data: ',training_data_precision)
print('Recall on Training data: ',training_data_recall)
print('F1 score on Training data: ',training_data_f1_score)


# In[169]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
test_data_precision = precision_score(X_test_prediction, Y_test)
test_data_recall = recall_score(X_test_prediction, Y_test)
test_data_f1_score = f1_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
print('precision score on Test Data:',test_data_precision)
print('Recall score on Test Data:',test_data_recall)
print('F1 score on Test Data:',test_data_f1_score)


# In[170]:


from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier object
model = DecisionTreeClassifier()

# Train the decision tree classifier with Training Data
model.fit(X_train, Y_train)


# In[171]:


# Check the dimensions of X_train and Y_train
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# In[172]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
training_data_precision = precision_score(X_train_prediction, Y_train)
training_data_recall = recall_score(X_train_prediction, Y_train)
training_data_f1_score = f1_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
print('precision on Training data: ',training_data_precision)
print('Recall on Training data: ',training_data_recall)
print('F1 score on Training data: ',training_data_f1_score)


# In[173]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
test_data_precision = precision_score(X_test_prediction, Y_test)
test_data_recall = recall_score(X_test_prediction, Y_test)
test_data_f1_score = f1_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
print('precision score on Test Data:',test_data_precision)
print('Recall score on Test Data:',test_data_recall)
print('F1 score on Test Data:',test_data_f1_score)


# In[174]:


# Check the dimensions of X_train and Y_train
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# In[175]:


from sklearn.svm import SVC

# Create an SVM classifier object
svm_classifier = SVC()

# Train the SVM classifier with Training Data
svm_classifier.fit(X_train, Y_train)


# In[176]:


X_train_prediction = svm_classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
training_data_precision = precision_score(X_train_prediction, Y_train)
training_data_recall = recall_score(X_train_prediction, Y_train)
training_data_f1_score = f1_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
print('precision on Training data: ',training_data_precision)
print('Recall on Training data: ',training_data_recall)
print('F1 score on Training data: ',training_data_f1_score)


# In[177]:


X_test_prediction = svm_classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
test_data_precision = precision_score(X_test_prediction, Y_test)
test_data_recall = recall_score(X_test_prediction, Y_test)
test_data_f1_score = f1_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
print('precision score on Test Data:',test_data_precision)
print('Recall score on Test Data:',test_data_recall)
print('F1 score on Test Data:',test_data_f1_score)


# In[181]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Generate example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'SVDD': OneClassSVM(kernel='rbf', nu=0.1)
}

# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Get predicted probabilities for the positive class
    if model_name == 'SVDD':
        y_train_probabilities = model.decision_function(X_train_scaled)
        y_test_probabilities = model.decision_function(X_test_scaled)
    else:
        y_train_probabilities = model.predict_proba(X_train_scaled)[:, 1]
        y_test_probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate ROC curve and AUC score
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probabilities)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probabilities)
    auc_train = roc_auc_score(y_train, y_train_probabilities)
    auc_test = roc_auc_score(y_test, y_test_probabilities)
    
    # Plot ROC curve
   
    plt.plot(fpr_test, tpr_test, linestyle='--', label=f'{model_name}') 

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Set plot labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Models (Before Feature Selection)')
plt.legend()
plt.grid(True)
plt.show()


# In[180]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Generate some example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
columns = [f'v{i}' for i in range(1, 21)]
df = pd.DataFrame(X, columns=columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# SelectKBest feature selection
# Here, we use f_classif score function for classification
k_best = SelectKBest(score_func=f_classif, k=17)  # Select top 17 features
X_train_kbest = k_best.fit_transform(X_train, y_train)
selected_features_kbest = df.columns[k_best.get_support()]

# Recursive Feature Elimination (RFE) with Random Forest
rf = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=14, step=1)  # Select top 14 features
X_train_rfe = rfe.fit_transform(X_train, y_train)
selected_features_rfe = df.columns[rfe.support_]

# Embedded Method using Random Forest
rf_embedded = RandomForestClassifier(random_state=42)
rf_embedded.fit(X_train, y_train)
importance = rf_embedded.feature_importances_
selected_features_embedded = df.columns[np.argsort(importance)[::-1][:14]]  # Select top 14 features based on importance

# Print selected features for each method
print("Selected features using SelectKBest:", selected_features_kbest)
print("Selected features using RFE:", selected_features_rfe)
print("Selected features using Embedded Method:", selected_features_embedded)


# In[ ]:




