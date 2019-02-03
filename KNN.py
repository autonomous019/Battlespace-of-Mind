
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[3]:

df = pd.read_csv('Classified Data',index_col=0)


# In[4]:

df.head()


# In[5]:

from sklearn.preprocessing import StandardScaler


# In[6]:

scaler = StandardScaler()


# In[7]:

scaler.fit(df) #fit scaler to the data


# In[8]:

sclaer.fit(df.drop('TARGET CLASS',axis=1))


# In[9]:

scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[10]:

#do a tranformation 
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1)


# In[11]:

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))


# In[12]:

scaled_features


# In[13]:

#create a features dataframe
df_feat = pd.DataFrame(scaled_features,columns=df.columns) #data is the scaled features


# In[14]:

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[15]:

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[16]:

df_feat.head()


# In[17]:

from sklearn.cross_validation import train_test_split


# In[18]:

X = df_feat
y = df['TARGET_CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[19]:

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30,random_state=101)


# In[20]:

from sklearn.neighbors import KNeighborsClassifier


# In[22]:

knn = KNeighborsClassifier(n_neighbors=1)


# In[23]:

knn.fit(X_train,y_train)


# In[24]:

pred = knn.predict(X_test)


# In[25]:

pred


# In[26]:

from sklearn.metrics import classification_report,confusion_matrix


# In[27]:

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[28]:

#to get a optimized K value, (nodes) use the elbow method
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test)) #append the mean of predictions not equal to the actual test values


# In[29]:

plt.figure(figsize=(10,6))


# In[32]:

plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[33]:

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:



