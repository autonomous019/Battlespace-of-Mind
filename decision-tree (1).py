
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[3]:

df = pd.read_csv('kyphosis.csv')


# In[4]:

df.head()


# In[5]:

sns.pairplot(df,hue='Kyphosis')


# In[6]:

from sklearn.cross_validation imort train_test_split


# In[7]:

from sklearn.cross_validation import train_test_split


# In[8]:

X = df.drop('Kyphosis',axis=1)


# In[9]:

y = df['Kyphosis']


# In[10]:

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[11]:

#train a single decision tree
from sklearn.tree import DecisionTreeClassifier


# In[12]:

dtree = DecisionTreeClassifier()


# In[13]:

dtree.fit(X_train,y_train)


# In[14]:

predictions = dtree.predict(X_test)


# In[15]:

from sklearn.metrics import classification_report, confusion_matrix


# In[16]:

print(confusion_matrix)


# In[17]:

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


# In[18]:

#now compare these results to a random forest model
from sklearn.ensemble import RandomForestClassifier


# In[19]:

#random forest are decision trees, an ensemble of them
rfc = RandomForestClassifier(n_estimators=200)


# In[20]:

rfc.fit(X_train,y_train)


# In[21]:

rfc_pred = rfc.predict(X_test)


# In[22]:

print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))
#train data vs. test data


# In[23]:

#due you value precison or recall; absent or present
#probably more important to be present, the attribute analysed
#random forest performs better then one decision tree
df['Kyphosis'].value_counts()


# In[24]:

#alot more absent vs. present for Kyphosis, the attribute analysed
#tree visualisation helps understanding how the tree metrics operate
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns[1:])
features


# In[26]:

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


# In[27]:

loans = pd.read_csv('loan_data.csv')


# In[28]:

loans.info()


# In[29]:

loans.describe()


# In[30]:

loans.head()


# In[31]:

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[32]:

#notfully paid column now
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[33]:

#create a countplot showing the counts of loans by purpose, hue is not.fully.paid
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# In[34]:

#now the trend between FICO score and interest rate
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[35]:

#lmpolots examine trend between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[36]:

#now set up our data for our Random Forest classification model
loans.info()


# In[37]:

#CATEGORICAL FEATURES, use dummy variables
cat_feats = ['purpose'] #create a string 'purpose' call this list cat_feats


# In[38]:

#use pd.get_dummies to create a fixed larger dataframe that has new feature cols with dummy vars
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[39]:

final_data.info()


# In[40]:

#split data into training and test sets
from sklearn.model_selection import train_test_split


# In[41]:

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[42]:

#train a single decision tree first
from sklearn.tree import DecisionTreeClassifier


# In[43]:

#create a instance of dec tree classifier then fit it to the training data
dtree = DecisionTreeClassifier()


# In[44]:

dtree.fit(X_train,y_train)


# In[45]:

#now the predictions and evaluations of decision tree
predictions = dtree.predict(X_test)


# In[46]:

from sklearn.metrics import classification_report,confusion_matrix


# In[47]:

print(classification_report(y_test,predictions))


# In[48]:

print(confusion_matrix(y_test,predictions))


# In[49]:

#train the random forest model
#instance of randomForestClassifier and fit the training data
from sklearn.ensemble import RandomForestClassifier


# In[50]:

#note size of n_estimators, could need adjusting, something to experiment with
rfc = RandomForestClassifier(n_estimators=600)


# In[51]:

#fit to training 
rfc.fit(X_train,y_train)


# In[52]:

#predictions and evaluations
predictions = rfc.predict(X_test)


# In[53]:

from sklearn.metrics import classification_report,confusion_matrix


# In[54]:

print(classification_report(y_test,predictions))


# In[55]:

print(confusion_matrix(y_test,predictions))


# In[56]:

#random forest performs better as is usually the case


# In[ ]:



