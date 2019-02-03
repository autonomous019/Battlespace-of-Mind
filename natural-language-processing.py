
# coding: utf-8

# In[1]:

import nltk


# In[2]:

import pandas as pd


# In[3]:

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))


# In[4]:

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')


# In[5]:

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


# In[6]:

messages.groupby('label').describe()


# In[7]:

messages['length'] = messages['message'].apply(len)
messages.head()


# In[8]:

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[9]:

messages['length'].plot(bins=50, kind='hist') 


# In[10]:

messages.length.describe()


# In[11]:

messages[messages['length'] == 910]['message'].iloc[0]


# In[12]:

messages.hist(column='length', by='label', bins=50,figsize=(12,4))


# In[13]:

import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)


# In[14]:

from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[15]:

nltk.download('stopwords')


# In[16]:

from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[17]:

nopunc.split()


# In[18]:

# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[19]:

clean_mess


# In[20]:

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[21]:

messages.head()


# In[22]:

# Check to make sure its working
messages['message'].head(5).apply(text_process)


# In[23]:

# Show original dataframe
messages.head()


# In[24]:

from sklearn.feature_extraction.text import CountVectorizer


# In[25]:

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[26]:

message4 = messages['message'][3]
print(message4)


# In[27]:

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# In[28]:

print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])


# In[29]:

messages_bow = bow_transformer.transform(messages['message'])


# In[30]:

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[31]:

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[32]:

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# In[33]:

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# In[34]:

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[35]:

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[36]:

print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])


# In[37]:

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[38]:

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))


# In[39]:

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[40]:

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[41]:

pipeline.fit(msg_train,label_train)


# In[42]:

predictions = pipeline.predict(msg_test)


# In[43]:

print(classification_report(predictions,label_test))


# In[ ]:



