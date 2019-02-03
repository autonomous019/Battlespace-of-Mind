
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

df = pd.read_csv('Ecommerce_Customers.csv')


# In[5]:

df.head()


# In[6]:

df.info()


# In[7]:

df.describe()


# In[8]:

df.columns


# In[9]:

sns.jointplot(df)


# In[10]:

#create a jointplot to compare the time on website and yearly amound spent
df['Time on Website']


# In[11]:

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)


# In[12]:

sns.jointplot(x='Time on App', y='Yearly Amound Spent', data=df)


# In[13]:

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)


# In[14]:

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df)


# In[15]:

sns.pairplot(df)


# In[16]:

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)


# In[17]:

y = df['Yearly Amount Spent']


# In[18]:

X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[19]:

from sklearn.cross_validation import train_test_split


# In[20]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[21]:

from sklearn.linear_model import LinearRegression


# In[22]:

lm = LinearRegression()


# In[23]:

lm.fit(X_train, y_train)


# In[24]:

print(lm.intercept_)


# In[25]:

lm.coef_


# In[26]:

predictions = lm.predict(X_test)


# In[27]:

predictions


# In[28]:

plt.scatte(y_test, predictions)


# In[29]:

plt.scatter(y_test, predictions)


# In[30]:

sns.distplot(y_test-predictions)


# In[31]:

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[32]:

sns.distplot((y_test-predictions),bins=50);


# In[ ]:



