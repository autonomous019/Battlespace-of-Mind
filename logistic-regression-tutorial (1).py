
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

train = pd.read_csv('titanic_train.csv')


# In[4]:

train.head()


# In[5]:

#see missing data
train.isnull()


# In[6]:

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[7]:

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[8]:

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[9]:

sns.set_style('whitegrid')


# In[10]:

sns.countplot(x='Survived',data=train)


# In[11]:

get_ipython().magic('matplotlib inline')


# In[12]:

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[13]:

sns.countplot(x='Survived',data=train)


# In[14]:

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[15]:

sns.countplot(x='Survived',hue='Pclass',data=train)


# In[16]:

train['Fare'].hist


# In[17]:

train['Fare'].hist()


# In[18]:

import cufflinks as cf


# In[19]:

import cufflinks as cf


# In[20]:

train.drop('Cabin',axis=1,inplace=True)


# In[21]:

train.head()


# In[22]:

train.dropna(inplace=True)


# In[23]:

#convert categorical features into dummy variables using pandas, convert binary values to numerical values like male/femal to 0,1
pd.get_dummies(train['Sex'])


# In[24]:

sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[25]:

sex.head()


# In[26]:

embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[27]:

embark.head()


# In[28]:

train = pd.concat([train,sex,embark],axis=1)


# In[29]:

train.head()


# In[30]:

#converted text to numerical or 0,1 variables


# In[31]:

#drop unused columnes
train.drop(['Sex','Embarked','Name','Ticket'], axis=1,inplace=True)


# In[32]:

train.head()


# In[33]:

#drop passengerID since it has no bearing on real analysis
train.drop('PassengerId',axis=1,inplace=True)


# In[34]:

train.head()


# In[35]:

#note that there is a difference between treating a column as a category and as a dummy variable, ie.Pclass could be recast as a dummy variable


# In[36]:

#you would now need to clean the test data the same way as the train data


# In[37]:

X = train.drop('Survived',axis=1)
y = train['Survived'] #is the column your trying to predict, the label to be applied


# In[38]:

from sklearn.cross_validation import train_test_split


# In[39]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[40]:

#just split data into a test set and a training set


# In[41]:

from sklearn.linear_model import LogisticRegression


# In[43]:

logmodel = LogisticRegression()


# In[44]:

#fit the model
logmodel.fit(X_train,y_train)


# In[45]:

predictions = logmodel.predict(X_test)


# In[46]:

#we have created a model, fit a model and predicted a model


# In[47]:

#most of the work is on the cleaning the data set, cleaning and more cleaning


# In[48]:

#classification tasks
from sklearn.metrics import classification_report


# In[49]:

#report tells precision, f1 score, accuracy, etc
print(classification_report(y_test, predictions))


# In[50]:

#you can also get the confusion matrix
from sklearn.metrics import confusion_matrix


# In[51]:

confusion_matrix(y_test,predictions)


# In[52]:

#to increase the precision, recall or accuracy try using the entirety of the training set vs test.csv, more feature engineering grab the title of the name as a feature or maybe the cabin letter could be a feature, or ticket all these could have more information and all information has value


# In[53]:

#see kaggle for more trailheads


# In[ ]:



