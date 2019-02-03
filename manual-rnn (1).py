
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# Number of inputs for each example
num_inputs = 2

# Number of neurons in first layer
num_neurons = 3


# In[3]:

# We now need two Xs! One for each timestamp (t=0 and t=1)
x0 = tf.placeholder(tf.float32,[None,num_inputs])
x1 = tf.placeholder(tf.float32,[None,num_inputs])


# In[4]:

# We'll also need a Weights variable for each x
# Notice the shape dimensions on both!
Wx = tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons]))
Wy = tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))


# In[5]:

b = tf.Variable(tf.zeros([1,num_neurons]))


# In[6]:

# First Activation
y0 = tf.tanh(tf.matmul(x0,Wx) + b)
y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) + b)


# In[7]:

init = tf.global_variables_initializer()


# In[8]:

# BATCH 0:       example1 , example2, example 3
x0_batch = np.array([[0,1],  [2,3],    [4,5]]) # DATA AT TIMESTAMP = 0

# BATCH 0:          example1 ,   example2,   example 3
x1_batch = np.array([[100,101], [102,103],  [104,105]]) # DATA AT TIMESTAMP = 1


# In[9]:

with tf.Session() as sess:
    
    sess.run(init)
    
    y0_output_vals , y1_output_vals  = sess.run([y0,y1],feed_dict={x0:x0_batch,x1:x1_batch})


# In[10]:

# The output of values at t=0
y0_output_vals


# In[11]:

# Output at t=1
y1_output_vals


# In[ ]:



