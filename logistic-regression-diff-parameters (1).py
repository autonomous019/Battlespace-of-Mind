
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

get_ipython().magic('matplotlib inline')


# In[4]:

train = pd.read_csv('advertising.csv')


# In[5]:

train.head()


# In[6]:

ad_data = pd.read_csv("advertising.csv")


# In[7]:

ad_data.head()


# In[8]:

ad_data.describe()


# In[9]:

ad_data['Age'].hist()


# In[10]:

sns.jointplot(x='Area Income', y='Age', data=ad_data)


# In[11]:

sns.jointplot(x='Daily Time Spent on Site', y='Age',kind=kde,data=ad_data)


# In[12]:

sns.jointplot(x='Daily Time Spent on Site', y='Age',kind='kde',data=ad_data)


# In[13]:

sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage',data=ad_data)


# In[14]:

sns.pairplot(data=ad_data,hue='Clicked on Ad')


# In[15]:

ad_data.drop('Add Topic Line',axis=1,inplace=True)


# In[16]:

ad_data.columns


# In[17]:

ad_data.head()


# In[18]:

ad_data.drop(['Ad Topic Line','City','Country','Timestamp'], axis=1,inplace=True)


# In[19]:

ad_data.head()


# In[20]:

X = ad_data.drop('Age',axis=1)
y = ad_data['Age'] #is the column your trying to predict, the label to be applied


# In[21]:

from sklearn.cross_validation import train_test_split


# In[22]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[23]:

from sklearn.linear_model import LogisticRegression


# In[24]:

logmodel = LogisticRegression()


# In[25]:

logmodel.fit(X_train,y_train)


# In[26]:

predictions = logmodel.predict(X_test)


# In[27]:

from sklearn.metrics import classification_report


# In[28]:

print(classification_report(y_test, predictions))


# In[29]:

from sklearn.metrics import confusion_matrix


# In[30]:

confusion_matrix(y_test,predictions)


# In[31]:

X = ad_data.drop('Male',axis=1)
y = ad_data['Male'] #is the column your trying to predict, the label to be applied


# In[32]:

from sklearn.cross_validation import train_test_split


# In[33]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[34]:

from sklearn.linear_model import LogisticRegression


# In[35]:

logmodel = LogisticRegression()


# In[36]:

logmodel.fit(X_train,y_train)


# In[37]:

predictions = logmodel.predict(X_test)


# In[38]:

from sklearn.metrics import classification_report


# In[39]:

print(classification_report(y_test, predictions))


# In[40]:

from sklearn.metrics import confusion_matrix


# In[41]:

confusion_matrix(y_test,predictions)


# In[ ]:



