
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

print(cancer['feature_names'])


# In[8]:

#with principle component analysis we are looking for what components explain the most variance with the dataset
df = pd.DataFrame(cancer['data'],columnss=cancer['feature_names'])


# In[9]:

#with principle component analysis we are looking for what components explain the most variance with the dataset
df = pd.DataFrame(cancer['data'],columnss=cancer['feature_names'])


# In[10]:

#with principle component analysis we are looking for what components explain the most variance with the dataset
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])


# In[11]:

df.head()


# In[12]:

from sklearn.preprocessing import StandardScaler


# In[13]:

scaler = StandardScaler()
scaler.fit(df)


# In[14]:

scaled_data = scaler.transform(df)


# In[15]:

from sklearn.decomposition import PCA


# In[16]:

pca = PCA(n_components=2)


# In[17]:

pca.fit(scaled_data)


# In[18]:

x_pca = pca.transform(scaled_data)


# In[19]:

scaled_data.shape


# In[20]:

x_pca.shape


# In[21]:

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[22]:

pca.components_


# In[23]:

df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[24]:

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# In[25]:

#so now what you could do is a logistic regression model on the x_pca data now that it is just two components, or even use a SVM model. 


# In[ ]:



