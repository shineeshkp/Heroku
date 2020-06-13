#!/usr/bin/env python
# coding: utf-8

# ### Classification of glasses

# #### KNN Steps:
#     1. Load the data
#     2. Initialize the value of k
#     3. For getting the predicted class, iterate from 1 to total # of training data
#     4. Calculate the distance between each test data and each row of trainign data. Hence we will use euclidean distance as our distance metric since its the most popular method. Other metrics that can be used are Chebyshev, cosine atc. 
#     5. Sort the calu=culated distances in acsebding order based on distance values
#     6. Get top k rows from the sorted array
#     7. Get the most frequent class for classification/mean or meadian foe regression of rows
#     8. Return preicted class
#     

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle


# In[2]:


glass = pd.read_csv("glass.csv")
glass.head()


# ### Standardize the data

# #### Compute the skewness in data

# In[13]:


#glass.skew() # Gives the measure of how much skew is there


# Standardization brings all data into the same unit. **After standardization** of data the **distribution of data (skew) should not be changed**.

# In[14]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()


# In[15]:


label = glass.pop('Type')


# In[16]:


sc.fit(glass) # Fit the data to standard scaler


# In[17]:


glass_scale= sc.transform(glass) # Transform the data
pd.DataFrame(glass_scale, columns=glass.columns)


# ### Train Test split

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


data_train, data_test, label_train, label_test = train_test_split(glass_scale, label, test_size=0.3, random_state =42)


# In[21]:


print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)


# ### Modelling and Prediction

# #### Instantiate KNN Classifier

# In[22]:


from sklearn.neighbors import KNeighborsClassifier


# In[23]:


knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski', n_jobs=1)


# In[24]:


knn.fit(data_train,label_train)


# In[25]:


y_preds= knn.predict(data_test)


# In[31]:


y_preds


pickle.dump(knn, open('model.pkl','wb'))

model=pickle.load(open('model.pkl', 'rb'))
print(model.predict(data_test[1].reshape(1,-1)))
print(model.predict(data_test[2].reshape(1,-1)))
print(model.predict(data_test[10].reshape(1,-1)))
print(model.predict(data_test[-1].reshape(1,-1)))

