#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[66]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')


# In[67]:


# first 5 rows of the dataset
credit_card_data.head()


# In[68]:


def hist(x):
    plt.hist(credit_card_data[x], bins=25)
    plt.title(x, fontsize=10, loc="right")
    plt.xlabel('Relative frequency')
    plt.ylabel('Absolute frequency')
    plt.show()


# In[69]:


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


# In[70]:


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


# In[10]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[11]:


print(legit.shape)
print(fraud.shape)


# In[12]:


# statistical measures of the data
legit.Amount.describe()
     


# In[13]:


fraud.Amount.describe()


# In[14]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[15]:


legit_sample = legit.sample(n=492)


# In[16]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)
     

new_dataset.head()


# In[17]:


new_dataset.tail()


# In[18]:


new_dataset['Class'].value_counts()


# In[19]:


new_dataset.groupby('Class').mean()


# In[20]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
     


# In[21]:


print(X)


# In[22]:


print(Y)


# In[23]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Randomly sample the same number of legitimate transactions as fraudulent transactions
legit_sample = legit.sample(n=492)

# Concatenate the sampled legitimate transactions with the fraudulent transactions
new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
     

print(X.shape, X_train.shape, X_test.shape)


# In[25]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the model to your training data
rf_classifier.fit(X_train, Y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()


# In[26]:


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV

# Generate example data
X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the search space for hyperparameters
param_space = {'C': (1e-6, 1e+6, 'log-uniform')}

# Initialize the Bayesian Optimization search
optimizer = BayesSearchCV(
    estimator=LogisticRegression(),
    search_spaces=param_space,
    scoring='accuracy',
    cv=5,
    n_iter=10,  # Number of iterations (trials)
    random_state=42
)

# Perform hyperparameter optimization
optimizer.fit(X_train_scaled, y_train)

# Get the best hyperparameters found during optimization
best_params = optimizer.best_params_

# Train the final model with the best hyperparameters
final_model = LogisticRegression(**best_params)
final_model.fit(X_train_scaled, y_train)

# Evaluate the final model
y_pred_train = final_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
y_pred_test = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Best Hyperparameters:", best_params)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[27]:


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV

# Generate example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the search space for hyperparameters
param_space = {'C': (1e-6, 1e+6, 'log-uniform')}

# Initialize the Bayesian Optimization search
optimizer = BayesSearchCV(
    estimator=LogisticRegression(),
    search_spaces=param_space,
    scoring='accuracy',
    cv=5,
    n_iter=10,  # Number of iterations (trials)
    random_state=42
)

# Perform hyperparameter optimization
optimizer.fit(X_train_scaled, y_train)

# Get the best hyperparameters found during optimization
best_params = optimizer.best_params_

# Train the final model with the best hyperparameters
final_model = LogisticRegression(**best_params)
final_model.fit(X_train_scaled, y_train)

# Evaluate the final model
y_pred_train = final_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
y_pred_test = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Best Hyperparameters:", best_params)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[28]:


import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import pyswarms as ps

# Generate example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def forward_prop(params):
    n_inputs = 20  # Update number of input features
    n_hidden = 20
    n_classes = 2

    W1 = params[0:400].reshape((n_inputs, n_hidden))  # Update slicing indices
    b1 = params[400:420].reshape((n_hidden,))
    W2 = params[420:460].reshape((n_hidden, n_classes))
    b2 = params[460:462].reshape((n_classes,))

    z1 = X_scaled.dot(W1) + b1
    a1 = np.tanh(z1)     
    z2 = a1.dot(W2) + b2
    logits = z2          

    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = len(X)
    corect_logprobs = -np.log(probs[range(N), y])
    loss = np.sum(corect_logprobs) / N
    return loss

# Define fitness function for PL-PSO
def f(x):
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

# Set dimensions
dimensions = 462  # Update dimensions to match the size of the parameter vector

# Define PSO parameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

# Initialize PL-PSO optimizer
optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=150, verbose=3)


# In[29]:


pos


# In[30]:


from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)


# In[31]:


plot_cost_history(cost_history=optimizer.cost_history)
plt.show()


# In[32]:


## Histograms
fig = plt.figure(figsize=(15, 20))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(credit_card_data.shape[1]):
    plt.subplot(8, 4, i + 1)
    f = plt.gca()
    f.set_title(credit_card_data.columns.values[i])

    vals = np.size(credit_card_data.iloc[:, i].unique())
    if vals >= 100:
        vals = 100                                    # limit our bins to 100 maximum
    
    plt.hist(credit_card_data.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[64]:


import pandas as pd
import numpy as np
import keras
from matplotlib import pyplot as plt


np.random.seed(2)




data = pd.read_csv('creditcard.csv')





data.head()



len(data)





data.describe()



fig = plt.figure(figsize=(15, 20))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(data.shape[1]):
    plt.subplot(8, 4, i + 1)
    f = plt.gca()
    f.set_title(data.columns.values[i])

    vals = np.size(data.iloc[:, i].unique())
    if vals >= 100:
        vals = 100                                    # limit our bins to 100 maximum
    
    plt.hist(data.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])




print('Number of fraudulent transactions = %d or %d per 100,000 transactions in the dataset'
      %(len(data[data.Class==1]), len(data[data.Class==1])/len(data)*100000))



data2 = data.drop(columns = ['Class'])   # drop non numerical columns
data2.corrwith(data.Class).plot.bar(
        figsize = (20, 10), title = "Correlation with Class Fraudulent or Not", fontsize = 15,
        rot = 45, grid = True)
plt.show()



from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))  # Normalize 'Amount' in [-1,+1] range
data = data.drop(['Amount'],axis=1)




data.head()




data = data.drop(['Time'],axis=1)
data.head()




X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']  # Response variable determining if fraudulent or not





y.head()





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)





X_train.shape





X_test.shape





from sklearn.ensemble import RandomForestClassifier




random_forest = RandomForestClassifier(n_estimators=100)





random_forest.fit(X_train,y_train.values.ravel())    # np.ravel() Return a contiguous flattened array





y_pred = random_forest.predict(X_test)


# In[21]:


random_forest.score(X_test,y_test)


# In[22]:



import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[23]:


# Confusion matrix on the test dataset
cnf_matrix = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# - while only 6 regular transactions are wrongly predicted as fraudulent, the model only detects 78% of the fraudulent transactions. As a consequence 33 fraudulent transactions are not detected (False Negatives).
# - Let's see if we can improve this performance with other machine learning / deep learning models in the rest of the notebook.

# In[24]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, plot_roc_curve
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy:%0.4f'%acc,'\tprecision:%0.4f'%prec,'\trecall:%0.4f'%rec,'\tF1-score:%0.4f'%f1)


# Let's store each model's performance in a dataframe for comparison purpose

# In[25]:


### Store results in dataframe for comparing various Models
results_testset = pd.DataFrame([['RandomForest', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset


# In[26]:


ROC_RF = plot_roc_curve(random_forest, X_test, y_test)
plt.show()


# We will run the models on the full dataset to check.

# In[27]:


# Confusion matrix on the whole dataset
y_pred = random_forest.predict(X)
cnf_matrix = confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[28]:


acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print('accuracy:%0.4f'%acc,'\tprecision:%0.4f'%prec,'\trecall:%0.4f'%rec,'\tF1-score:%0.4f'%f1)


# In[29]:


results_fullset = pd.DataFrame([['RandomForest', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset


# ## Decision trees

# In[30]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()


# In[31]:


decision_tree.fit(X_train,y_train.values.ravel())


# In[32]:


y_pred = decision_tree.predict(X_test)


# In[33]:


decision_tree.score(X_test,y_test)


# In[34]:


# Confusion matrix on the test dataset
cnf_matrix = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# - The performance of the Decision Tree model is below the one using Random Forest. Let's check the performance indicators.

# In[35]:


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[36]:


### Store results in dataframe for comparing various Models
model_results = pd.DataFrame([['DecisionTree', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset


# In[37]:


ROC_DT = plot_roc_curve(decision_tree, X_test, y_test)
plt.show()


# In[38]:


# Confusion matrix on the whole dataset
y_pred = decision_tree.predict(X)
cnf_matrix = confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[39]:


acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)


# In[40]:


model_results = pd.DataFrame([['DecisionTree', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset = results_fullset.append(model_results, ignore_index = True)
results_fullset


# ## Let's now explore Neural Network models

# In[41]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ## Deep neural network
# - We will use a simple NN made of 5 fully-connected layers with ReLu activation. The NN takes a vector of length 29 as input. This represents the information related to each transactions, ie each line with 29 columns from the dataset. For each transaction, the final layer will output a probability distribution (sigmoid activation function) and classify either as not fraudulent (0) or fraudulent (1).
# - a dropout step is included to prevent overfitting.

# In[42]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[43]:


model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),   # input of 29 columns as shown above
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(24,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),                        # binary classification fraudulent or not
])


# In[44]:


model.summary()


# ## Training

# In[45]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[46]:


score = model.evaluate(X_test, y_test)


# In[47]:


print(score)


# - The model achieves an accuracy of 99.94% ! Is this a good performance ?
# - Remember that our dataset is significantly composed of non fraudulent samples with only 172 fraudulent transactions per 100,000. Consequently, a model predicting every transaction as 'non fraudulent' would achieve 99.83% accuracy despite being unable to detect a single fraudulent case !

# In[48]:


## Confusion Matrix on unsee test set
import seaborn as sn
y_pred = model.predict(X_test)
for i in range(len(y_test)):
    if y_pred[i]>0.5:
        y_pred[i]=1 
    else:
        y_pred[i]=0
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# Detection of fraudulent transactions did not improve compared to the previous machine learning models.
# - 115 fraudulent transactions are detected as fraudulent by the model, yet 32 fraudulent transactions are not identified (false negative) which remains an issue. Our objective must be to detect as many fraudulent transactions as possible since these can have a huge negative impact.
# - 21 regular transactions are detected as potentially fraudulent by the model. These are false positive. This number is negligible.
# 
# Conclusion : We must find ways to further reduce the number of false negative.

# In[49]:


# Alternative approach to plot confusion matrix (from scikit-learn.org site)
y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)    # Pandas format required by confusion_matrix function


# In[50]:


cnf_matrix = confusion_matrix(y_test, y_pred.round())   # y_pred.round() to convert probability to either 0 or 1 in line with y_test


# In[51]:


print(cnf_matrix)


# In[52]:


plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[53]:


acc = accuracy_score(y_test, y_pred.round())
prec = precision_score(y_test, y_pred.round())
rec = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())


# In[54]:


### Store results in dataframe for comparing various Models
model_results = pd.DataFrame([['PlainNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset


# In[55]:


# Confusion matrix on the whole dataset
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# In[56]:


acc = accuracy_score(y, y_pred.round())
prec = precision_score(y, y_pred.round())
rec = recall_score(y, y_pred.round())
f1 = f1_score(y, y_pred.round())


# In[57]:


model_results = pd.DataFrame([['PlainNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset = results_fullset.append(model_results, ignore_index = True)
results_fullset


# ### Weighted loss to account for large class imbalance in train dataset
# - we will adjust the class imbalance by giving additional weight to the loss associated to errors made on fraudulent transaction detection. Let's review the process:

# In[58]:


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), np.array([y_train[i][0] for i in range(len(y_train))]))
class_weights = dict(enumerate(class_weights))
class_weights


# - The class 'Fraudulent' (y=1) is assigned a weight of 289 vs 0.5 for the class 'not fraudulent' due to the very low prevalence we detected during data exploration. This allows the model to give more importance to the errors made on fraudulent cases during training.

# In[59]:


model.fit(X_train,y_train,batch_size=15,epochs=5, class_weight=class_weights, shuffle=True)


# In[60]:


score_weighted = model.evaluate(X_test, y_test)


# In[61]:


print(score_weighted)


# In[62]:


## Confusion Matrix on unseen test set
y_pred = model.predict(X_test)
for i in range(len(y_test)):
    if y_pred[i]>0.5:
        y_pred[i]=1 
    else:
        y_pred[i]=0
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# - The accuracy reduced a bit. We need to look at more detailed evaluation measures like precision and recall to gauge the true model performance.
# - The model is actually much better at detecting fraudulent cases now. We have a lower False negative rate which is the key criteria for our purpose (detect a fraud when there is one).
# - But on the other hand, the model generates an excessive number of false positive compared to the previous approaches. 

# In[63]:


acc = accuracy_score(y_test, y_pred.round())
prec = precision_score(y_test, y_pred.round())
rec = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())


# In[64]:


### Store results in dataframe for comparing various Models
model_results = pd.DataFrame([['WeightedNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset


# - Although the recall metric improves, the F1-score collapses due to extremely poor precision. Given the very high number of transactions processed, the excessive number of false positives is clearly an issue.

# In[65]:


# Confusion matrix on the whole dataset
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# In[66]:


acc = accuracy_score(y, y_pred.round())
prec = precision_score(y, y_pred.round())
rec = recall_score(y, y_pred.round())
f1 = f1_score(y, y_pred.round())


# In[67]:


model_results = pd.DataFrame([['WeightedNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset = results_fullset.append(model_results, ignore_index = True)
results_fullset


# ## Undersampling
# - In order to balance the train set, another technique is undersampling. With this technique, we adjust the largest class to match the number of samples of the under-represented class. Here we want to randomly pick an amount of non-fraudulent transactions equal to the number of fraudulent transactions in the dataset.

# In[68]:


fraud_indices = np.array(data[data.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)


# In[69]:


normal_indices = data[data.Class == 0].index


# In[70]:


len(normal_indices)


# In[71]:


# Random select N indices from non fraudulent samples (N equals to number of fraudulent records)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
print(len(random_normal_indices))


# In[72]:


under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))


# In[73]:


under_sample_data = data.iloc[under_sample_indices,:]


# In[74]:


X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)


# In[76]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[77]:


model.summary()




model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[79]:


y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()





acc = accuracy_score(y_test, y_pred.round())
prec = precision_score(y_test, y_pred.round())
rec = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())



model_results = pd.DataFrame([['UnderSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset





# Confusion matrix on the whole dataset
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()





acc = accuracy_score(y, y_pred.round())
prec = precision_score(y, y_pred.round())
rec = recall_score(y, y_pred.round())
f1 = f1_score(y, y_pred.round())




model_results = pd.DataFrame([['UnderSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset = results_fullset.append(model_results, ignore_index = True)
results_fullset





from imblearn.over_sampling import SMOTE





X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())




y_resample





print('Number of total transactions before SMOTE upsampling: ', len(y), '...after SMOTE upsampling: ', len(y_resample))
print('Number of fraudulent transactions before SMOTE upsampling: ', len(y[y.Class==1]), 
      '...after SMOTE upsampling: ', np.sum(y_resample[y_resample==1]))





y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)





X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)





X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)





model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),   # input of 29 columns as shown above
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(24,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),                        # binary classification fraudulent or not
])




model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)




y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()





acc = accuracy_score(y_test, y_pred.round())
prec = precision_score(y_test, y_pred.round())
rec = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())




### Store results in dataframe for comparing various Models
model_results = pd.DataFrame([['OverSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_testset = results_testset.append(model_results, ignore_index = True)
results_testset





# Confusion matrix on the whole dataset
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()




acc = accuracy_score(y, y_pred.round())
prec = precision_score(y, y_pred.round())
rec = recall_score(y, y_pred.round())
f1 = f1_score(y, y_pred.round())




model_results = pd.DataFrame([['OverSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
               columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
results_fullset = results_fullset.append(model_results, ignore_index = True)
results_fullset








# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')
     

# first 5 rows of the dataset
credit_card_data.head()
     


# In[ ]:


credit_card_data.tail()


# In[ ]:


# dataset informations
credit_card_data.info()


# In[ ]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[34]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[35]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
     

print(legit.shape)
print(fraud.shape)
     


# In[36]:


# statistical measures of the data
legit.Amount.describe()


# In[37]:


fraud.Amount.describe()


# In[38]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[39]:


legit_sample = legit.sample(n=492)


# In[40]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)
     

new_dataset.head()


# In[41]:


new_dataset.tail()


# In[42]:


new_dataset['Class'].value_counts()


# In[43]:


new_dataset.groupby('Class').mean()


# In[44]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
     

print(X)


# In[45]:


print(Y)


# In[46]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
     

print(X.shape, X_train.shape, X_test.shape)


# In[47]:


model = LogisticRegression()
     

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[48]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     

print('Accuracy on Training data : ', training_data_accuracy)


# In[49]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     

print('Accuracy score on Test Data : ', test_data_accuracy)


# In[50]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
Y_train_probabilities = model.predict_proba(X_train)[:, 1]
Y_test_probabilities = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, Y_train_probabilities)

# Calculate ROC curve for test data
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, Y_test_probabilities)

# Calculate AUC score
auc_score_train = roc_auc_score(Y_train, Y_train_probabilities)
auc_score_test = roc_auc_score(Y_test, Y_test_probabilities)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f'Training Data ROC Curve (AUC = {auc_score_train:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Test Data ROC Curve (AUC = {auc_score_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()


# In[51]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a Decision Tree classifier
decision_tree_model = DecisionTreeClassifier()

# Train the Decision Tree Model with Training Data
decision_tree_model.fit(X_train, Y_train)

# Accuracy on training data
train_predictions = decision_tree_model.predict(X_train)
training_accuracy = accuracy_score(train_predictions, Y_train)
print('Accuracy on Training data:', training_accuracy)

# Accuracy on test data
test_predictions = decision_tree_model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, Y_test)
print('Accuracy on Test Data:', test_accuracy)


# In[52]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
Y_train_probabilities = decision_tree_model.predict_proba(X_train)[:, 1]
Y_test_probabilities = decision_tree_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, Y_train_probabilities)

# Calculate ROC curve for test data
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, Y_test_probabilities)

# Calculate AUC score
auc_score_train = roc_auc_score(Y_train, Y_train_probabilities)
auc_score_test = roc_auc_score(Y_test, Y_test_probabilities)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f'Training Data ROC Curve (AUC = {auc_score_train:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Test Data ROC Curve (AUC = {auc_score_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Decision Tree')
plt.legend()
plt.grid(True)
plt.show()


# In[53]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an SVM classifier
svm_model = SVC()

# Train the SVM Model with Training Data
svm_model.fit(X_train, Y_train)

# Accuracy on training data
train_predictions = svm_model.predict(X_train)
training_accuracy = accuracy_score(train_predictions, Y_train)
print('Accuracy on Training data:', training_accuracy)

# Accuracy on test data
test_predictions = svm_model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, Y_test)
print('Accuracy on Test Data:', test_accuracy)


# In[54]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
Y_train_probabilities = svm_model.decision_function(X_train)
Y_test_probabilities = svm_model.decision_function(X_test)

# Calculate ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, Y_train_probabilities)

# Calculate ROC curve for test data
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, Y_test_probabilities)

# Calculate AUC score
auc_score_train = roc_auc_score(Y_train, Y_train_probabilities)
auc_score_test = roc_auc_score(Y_test, Y_test_probabilities)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f'Training Data ROC Curve (AUC = {auc_score_train:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Test Data ROC Curve (AUC = {auc_score_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM')
plt.legend()
plt.grid(True)
plt.show()


# In[55]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a Random Forest classifier
random_forest_model = RandomForestClassifier()

# Train the Random Forest Model with Training Data
random_forest_model.fit(X_train, Y_train)

# Accuracy on training data
train_predictions = random_forest_model.predict(X_train)
training_accuracy = accuracy_score(train_predictions, Y_train)
print('Accuracy on Training data:', training_accuracy)

# Accuracy on test data
test_predictions = random_forest_model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, Y_test)
print('Accuracy on Test Data:', test_accuracy)


# In[59]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
Y_train_probabilities = random_forest_model.predict_proba(X_train)[:, 1]
Y_test_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, Y_train_probabilities)

# Calculate ROC curve for test data
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, Y_test_probabilities)

# Calculate AUC score
auc_score_train = roc_auc_score(Y_train, Y_train_probabilities)
auc_score_test = roc_auc_score(Y_test, Y_test_probabilities)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f'Training Data ROC Curve (AUC = {auc_score_train:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Test Data ROC Curve (AUC = {auc_score_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend()
plt.grid(True)
plt.show()


# In[57]:


from sklearn.svm import OneClassSVM

# Create a One-Class SVM model
svdd_model = OneClassSVM(kernel='rbf', nu=0.1)  # Example parameters

# Fit the SVDD model to your training data (normal instances)
svdd_model.fit(X_train)

# Predict anomalies (abnormal instances) on training and test data
train_predictions = svdd_model.predict(X_train)
test_predictions = svdd_model.predict(X_test)

# Convert predictions (-1 for anomalies, 1 for normal instances) to binary labels (0 for anomalies, 1 for normal instances)
train_labels = (train_predictions == 1).astype(int)
test_labels = (test_predictions == 1).astype(int)

# Calculate accuracy on training and test data (accuracy is the proportion of correctly identified normal instances)
training_accuracy = (train_labels == 1).mean()
test_accuracy = (test_labels == 1).mean()

print('Accuracy on Training data:', training_accuracy)
print('Accuracy on Test Data:', test_accuracy)


# In[58]:


from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Assuming you have SVDD model already trained
svdd_model = OneClassSVM(kernel='rbf', nu=0.1)  # Example parameters

# Fit the SVDD model to your data (assuming X_train is your training data)
svdd_model.fit(X_train)

# Get decision scores for all data points (distance to the decision boundary)
decision_scores_train = svdd_model.decision_function(X_train)
decision_scores_test = svdd_model.decision_function(X_test)

# Calculate ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, decision_scores_train)

# Calculate ROC curve for test data
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, decision_scores_test)

# Calculate AUC score
auc_score_train = roc_auc_score(Y_train, decision_scores_train)
auc_score_test = roc_auc_score(Y_test, decision_scores_test)

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(fpr_test, tpr_test, label=f'Test Data ROC Curve ')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVDD')
plt.legend()
plt.grid(True)
plt.show()


# In[60]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Generate example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using filter method
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'SVDD': OneClassSVM(kernel='rbf', nu=0.1),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# Plot ROC curve for each model
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
plt.title('Receiver Operating Characteristic (ROC) Curve for Models with Filter Method')
plt.legend()
plt.grid(True)
plt.show()


# In[61]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Generate example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'SVDD': OneClassSVM(kernel='rbf', nu=0.1),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# Feature selection using wrapper method (Recursive Feature Elimination)
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use Decision Tree as the estimator for RFE
    rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10, step=1)
    X_train_selected = rfe_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = rfe_selector.transform(X_test_scaled)
    
    # Train the model
    model.fit(X_train_selected, y_train)
    
    # Get predicted probabilities for the positive class
    if model_name == 'SVDD':
        y_train_probabilities = model.decision_function(X_train_selected)
        y_test_probabilities = model.decision_function(X_test_selected)
    else:
        y_train_probabilities = model.predict_proba(X_train_selected)[:, 1]
        y_test_probabilities = model.predict_proba(X_test_selected)[:, 1]

    # Calculate ROC curve and AUC score
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probabilities)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probabilities)
    auc_train = roc_auc_score(y_train, y_train_probabilities)
    auc_test = roc_auc_score(y_test, y_test_probabilities)
    
    # Plot ROC curve
    
    plt.plot(fpr_test, tpr_test, linestyle='--', label=f'{model_name} ')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Set plot labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Models with Wrapper Method')
plt.legend()
plt.grid(True)
plt.show()


# In[62]:


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
    'Logistic Regression': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'SVDD': OneClassSVM(kernel='rbf', nu=0.1)
}

# Plot ROC curves after embedded feature selection
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
plt.title('Receiver Operating Characteristic (ROC) Curve for Models (After Embedded Feature Selection)')
plt.legend()
plt.grid(True)
plt.show()


# In[73]:


from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate some example data
X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=42)

# Train a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=[f'Feature {i}' for i in range(len(feature_importances))])
plt.title('Feature Importances of Random Forest with 100 Decision Trees')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.show()


# In[77]:


from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate some example data
X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=42)

# Train a random forest classifier with 50 trees
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

# Get feature importances for all trees
feature_importances = rf.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=[f'Feature {i}' for i in range(len(feature_importances))])
plt.title('Feature Importances of Random Forest with 50 Decision Trees')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.show()


# In[92]:


from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Train a random forest classifier with 50 trees
num_trees = 50
rf = RandomForestClassifier(n_estimators=num_trees, random_state=42)
rf.fit(X, y)

# Perform cross-validation to get accuracy scores for each tree
cv_scores = []
for tree_idx, tree in enumerate(rf.estimators_):
    scores = cross_val_score(tree, X, y, cv=5)  # 5-fold cross-validation
    cv_scores.append(scores.mean())

# Plot the accuracy scores of each tree
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_trees + 1), cv_scores, marker='o', linestyle='-')
plt.title('Accuracy of Decision Trees in Random Forest with 50 Decision Trees')
plt.xlabel('Tree Index')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# In[87]:


from sklearn.metrics import precision_recall_curve

# Initialize dictionaries to store precision-recall curves
precision_recall_curves = {}

# Calculate precision-recall curve for each model
for model_name, model in models.items():
    # Calculate predicted probabilities for positive class
    if model_name == 'SVDD':
        y_test_probabilities = model.decision_function(X_test_scaled)
    else:
        y_test_probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_probabilities)
    
    # Store precision-recall curve in dictionary
    precision_recall_curves[model_name] = (precision, recall)

# Plot ROC curve with precision for all algorithms
plt.figure(figsize=(10, 8))
for model_name, (precision, recall) in precision_recall_curves.items():
    plt.plot(recall, precision, label=model_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for All Algorithms')
plt.legend()
plt.grid(True)
plt.show()


# In[88]:


# Initialize dictionaries to store recall scores for all algorithms
recall_scores = {}

# Calculate recall score for each model
for model_name, model in models.items():
    # Calculate predicted labels
    if model_name == 'SVDD':
        y_test_pred = (model.decision_function(X_test_scaled) < 0).astype(int)  # Assuming -1 denotes anomalies
    else:
        y_test_pred = model.predict(X_test_scaled)
    
    # Calculate recall score
    recall_scores[model_name] = recall_score(y_test, y_test_pred)

# Plot recall scores for all algorithms
plt.figure(figsize=(10, 8))
for model_name, recall in recall_scores.items():
    plt.bar(model_name, recall, label=f'{model_name}: {recall:.2f}', alpha=0.7)

plt.xlabel('Algorithm')
plt.ylabel('Recall Score')
plt.title('Recall Scores for All Algorithms')
plt.legend()
plt.grid(True)
plt.show()


# In[89]:


# Find the algorithm with the highest recall score
best_algorithm_recall = max(recall_scores, key=recall_scores.get)
highest_recall = recall_scores[best_algorithm_recall]

print(f"The algorithm with the highest recall score is '{best_algorithm_recall}' with a recall score of {highest_recall:.2f}.")


# In[ ]:




