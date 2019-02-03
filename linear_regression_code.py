
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

df = pd.read_csv('USA_Housing.csv')


# In[5]:

df.head()


# In[6]:

df.describe()


# In[7]:

df.columns


# In[8]:

sns.pairplot(df)


# In[9]:

sns.distplot(df['Price']) #display average 'Price'


# In[10]:

sns.heatmap(df.corr()) #correlation matrix heat map


# In[11]:

sns.heatmap(df.corr(), annot=True) #gives actual values in heatmap


# In[12]:

#train a linear regression model
df.columns


# In[13]:

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[14]:

y = df['Price']


# In[15]:

#train test split of data
from sklearn.cross_validation import train_test_split


# In[16]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[17]:

#create and train the model
from sklearn.linear_model import LinearRegression


# In[18]:

#instantiate an instance of linear regression model
lm = LinearRegression()


# In[19]:

#fit the data
lm.fit(X_train, y_train)


# In[20]:

#evaluate our model, checking coefficients and see how we interpret them
print(lm.intercept_)


# In[21]:

#check coefficients, relate to each feature in datasset
lm.coef_


# In[22]:

#create a dataframe off of these results to better organize and analyse
X_train.columns


# In[23]:

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])


# In[24]:

cdf


# In[25]:

predictions = lm.predict(X_test)


# In[26]:

predictions


# In[27]:

y_test


# In[28]:

plt.scatte(y_test, predictions)


# In[29]:

plt.scatter(y_test, predictions)


# In[30]:

#create a histogram of our residual, the diff between the prediction values and actual values
sns.distplot(y_test-predictions)


# In[31]:

#if the plot is chaotic check to see if your model is off, it may not be best to use linear regression switch models
#regression evaluation metrics, there are 3 different evaluation metrics types
#Mean Absolute Error (MEA), is the mean of the absolute value of the errors
#Mean Squared Error (MSE), is the mean of the squared errors
#Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors
from sklearn import metrics


# In[32]:

#take in your y_test true and predictions
metrics.mean_absolute_error(y_test, predictions)


# In[33]:

metrics.mean_squared_error(y_test, predictions)


# In[34]:

#root mean squared error
np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:



