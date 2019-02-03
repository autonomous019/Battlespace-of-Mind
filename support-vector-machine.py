
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

from sklearn.datasets import load_breast_cancer


# In[3]:

cancer = load_breast_cancer()


# In[4]:

cancer.keys()


# In[5]:

print(cancer['DESCR'])


# In[6]:

cancer['feature_names']


# In[7]:

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()


# In[8]:

cancer['target']


# In[9]:

df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])


# In[10]:

df.head()


# In[11]:

from sklearn.model_selection import train_test_split


# In[12]:

X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)


# In[13]:

from sklearn.svm import SVC


# In[14]:

model = SVC()


# In[15]:

model.fit(X_train,y_train)


# In[16]:

predictions = model.predict(X_test)


# In[17]:

from sklearn.metrics import classification_report,confusion_matrix


# In[18]:

print(confusion_matrix(y_test,predictions))


# In[19]:

print(classification_report(y_test,predictions))


# In[20]:

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[21]:

from sklearn.model_selection import GridSearchCV


# In[22]:

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[23]:

grid.fit(X_train,y_train)


# In[24]:

grid.best_params_


# In[25]:

grid.best_estimator_


# In[26]:

grid_predictions = grid.predict(X_test)


# In[27]:

print(confusion_matrix(y_test,grid_predictions))


# In[28]:

print(classification_report(y_test,grid_predictions))


# In[29]:

#
#
#now trying on Iris data
#
#


# In[30]:

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[31]:

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[32]:

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# In[33]:

'''
3 classes of Iris dataset
    Iris-setosa, Iris-versicolor, Iris-virginica each n=50
four features
    sepal length in cm
    sepal width in cm
    petal length in cm
    petal width in cm
'''


# In[34]:

import seaborn as sns
iris = sns.load_dataset('iris')


# In[35]:

iris.head()


# In[36]:



sns.pairplot(df,hue='species')



# In[37]:



sns.pairplot(iris,hue='Kyphosis')



# In[38]:



sns.pairplot(iris,hue='species')



# In[39]:



sns.jointplot(x='sepal_length', y='sepal_width',kind='kde',data=iris)



# In[40]:

from sklearn.model_selection import train_test_split


# In[41]:

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[42]:

from sklearn.svm import SVC


# In[43]:

svc_model = SVC()


# In[44]:

svc_model.fit(X_train,y_train)


# In[45]:

predictions = svc_model.predict(X_test)


# In[46]:

from sklearn.metrics import classification_report,confusion_matrix


# In[47]:

print(confusion_matrix(y_test,predictions))


# In[48]:

print(classification_report(y_test,predictions))


# In[49]:

from sklearn.model_selection import GridSearchCV


# In[50]:

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# In[51]:

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=5)
grid.fit(X_train,y_train)


# In[52]:

grid_predictions = grid.predict(X_test)


# In[53]:

print(confusion_matrix(y_test,grid_predictions))


# In[54]:

print(classification_report(y_test,grid_predictions))


# In[ ]:



