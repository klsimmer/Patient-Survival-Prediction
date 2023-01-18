#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import (Dense,Activation)
from sklearn.model_selection import (train_test_split,
                                    RandomizedSearchCV,
                                    RepeatedStratifiedKFold,
                                    GridSearchCV
                                    )
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.callbacks import (History)
from sklearn.metrics import (accuracy_score, 
                            classification_report,
                            roc_auc_score, 
                            confusion_matrix,
                             mean_absolute_error,
                             mean_squared_error,
                             recall_score,
                             precision_score
                            )
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

from keras.regularizers import l2

from matplotlib import rcParams
rcParams['figure.figsize'] = 15, 5
sns.set_style('darkgrid')


# In[2]:


patient_df = pd.read_csv('dataset.csv')
patient_df.head()


# In[3]:


patient_df.describe()


# In[4]:


patient_df.info()


# In[5]:


patient_df.drop(labels=['apache_3j_bodysystem','apache_2_bodysystem'], axis=1, inplace=True)
patient_df = pd.get_dummies(data=patient_df, columns=['ethnicity', 'gender','icu_admit_source','icu_stay_type','icu_type'], drop_first=True)
patient_df.drop(labels=['Unnamed: 83','encounter_id', 'patient_id', 'hospital_id', 'icu_id'], axis=1, inplace=True)
patient_df = patient_df.dropna()


# In[6]:


patient_df.info()


# In[7]:




RandomForestModelData = patient_df.copy()
y = RandomForestModelData['hospital_death']
X = RandomForestModelData.drop('hospital_death', axis = 1)
# doing scalar
scaled_features = StandardScaler().fit_transform(X.values)
X = pd.DataFrame(scaled_features, columns=RandomForestModelData.loc[:,RandomForestModelData.columns != "hospital_death"].columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# In[ ]:


## DONT RUN UNLESS YOU WANT COMP TO BURN

# Optimize setting 
optimize = {'bootstrap': [True, False],
 'max_depth': [10,30,50],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 1000, 1800, ]}

RandomForestModel = RandomForestRegressor()

RandomForestModel_Optimizer = RandomizedSearchCV(estimator = RandomForestModel,param_distributions = optimize,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
RandomForestModel_Optimizer.fit(X_train, y_train)


# In[8]:


print ('Random grid: ', optimize, '\n')
# print the best parameters
print ('Best Parameters: ', RandomForestModel_Optimizer.best_params_, ' \n')


# In[ ]:


RandomForestModelOptimized = RandomForestRegressor()
RandomForestModelOptimized.fit( X_train, y_train) 


# In[ ]:


y_hat = RandomForestModel.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat))

print("Mean Absolute Error: ", mean_absolute_error(y_test, y_hat))
print("Mean Square Error: ", mean_squared_error(y_test, y_hat))
print("Accuracy Score: ", accuracy_score(y_test, y_hat))
print("Recall Score: ", recall_score(y_test, y_hat, average=None))
print("precision Score: ", precision_score(y_test, y_hat, average=None))
print("AUC-ROC Curve: ", roc_auc_score(y_test, y_hat))


# In[ ]:


group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')


# In[13]:


LogModedlData = patient_df.copy()
y = LogModedlData['hospital_death']
X = LogModedlData.drop('hospital_death', axis = 1)
# doing scalar
scaled_features = StandardScaler().fit_transform(X.values)
X = pd.DataFrame(scaled_features, columns=LogModedlData.loc[:,LogModedlData.columns != "hospital_death"].columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# In[15]:


# define models and parameters
LogModel = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=LogModel, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train,y_train )

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[20]:


LogModel = LogisticRegression(C = 1.0, penalty = "l2", solver= "newton-cg")
LogModel.fit(X_train,y_train)
y_hat = LogModel.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat))

print("Mean Absolute Error: ", mean_absolute_error(y_test, y_hat))
print("Mean Square Error: ", mean_squared_error(y_test, y_hat))
print("Accuracy Score: ", accuracy_score(y_test, y_hat))
print("Recall Score: ", recall_score(y_test, y_hat, average=None))
print("precision Score: ", precision_score(y_test, y_hat, average=None))
print("AUC-ROC Curve: ", roc_auc_score(y_test, y_hat))


# In[21]:


group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')


# In[22]:


NeuralNetworkData = patient_df.copy()
y = NeuralNetworkData['hospital_death']
X = NeuralNetworkData.drop('hospital_death', axis = 1)
# doing scalar
scaled_features = StandardScaler().fit_transform(X.values)
X = pd.DataFrame(scaled_features, columns=NeuralNetworkData.loc[:,NeuralNetworkData.columns != "hospital_death"].columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# In[25]:


NeuralNetwork = Sequential()
NeuralNetwork.add(Dense(91, input_dim=91, activation='relu'))
NeuralNetwork.add(Dense(60, activation='relu'))
NeuralNetwork.add(Dense(1,activation='sigmoid'))

opt = tf.keras.optimizers.SGD(learning_rate=0.00001)
NeuralNetwork.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = NeuralNetwork.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=50, batch_size=64)


# In[26]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[27]:


pd.DataFrame(history.history).plot(figsize=(12,8))
plt.show()


# In[28]:


history_dict = history.history
# Learning curve(Loss)
# let's see the training and validation loss by epoch

# loss
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this

# range of X (no. of epochs)
epochs = range(1, len(loss_values) + 1) 

# plot
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[29]:


# Learning curve(accuracy)
# let's see the training and validation accuracy by epoch

# accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# orange is for "orange"
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# this is the max value - should correspond to
# the HIGHEST train accuracy
np.max(val_acc)


# In[36]:


y_hat = NeuralNetwork.predict(X_test)
y_hat = np.where(y_hat>0.5,1,0)


# In[37]:


cf_matrix = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat))

print("Mean Absolute Error: ", mean_absolute_error(y_test, y_hat))
print("Mean Square Error: ", mean_squared_error(y_test, y_hat))
print("Accuracy Score: ", accuracy_score(y_test, y_hat))
print("Recall Score: ", recall_score(y_test, y_hat, average=None))
print("precision Score: ", precision_score(y_test, y_hat, average=None))
print("AUC-ROC Curve: ", roc_auc_score(y_test, y_hat))


# In[38]:


group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')


# In[ ]:




