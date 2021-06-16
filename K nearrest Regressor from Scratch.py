#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor 


# In[9]:


class K_Nearest_Neighbors_Regressor():
    def __init__( self, K ):
        self.K = K
    def fit(self,X_train,Y_train):
        self.X_train = X_train 
        self.Y_train = Y_train
        self.m, self.n = X_train.shape
    def predict(self, X_test):
        self.X_test = X_test 
        self.m_test, self.n = X_test.shape
        Y_predict = np.zeros(self.m_test)
        for i in range(self.m_test ):
            x = self.X_test[i]
            neighbors =np.zeros(self.K)
            neighbors = self.find_neighbors(x)
            Y_predict[i] = np.mean(neighbors)
        return Y_predict
    def find_neighbors(self,x):
        euclidean_distances = np.zeros(self.m)
        for i in range(self.m):
            d = self.euclidean(x,self.X_train[i])
            euclidean_distances[i]=d 
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K] 
    def euclidean(self,x,x_train):
        return np.sqrt( np.sum( np.square(x - x_train))) 
    


# In[6]:


df = pd.read_csv( "C:/Users/user/Desktop/IVY WORK BOOK/MACHINE LEARNING/Python Datasets/Regression Datasets/BikeRentData.csv")


# In[7]:


df.head(20)


# In[12]:


X = df.iloc[:,:-1].values 
Y = df.iloc[:,-1:].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0) 
#model = K_Nearest_Neighbors_Regressor(K=5)
#model.fit( X_train, Y_train) 
Y_pred = model.predict(X_test)
dftest=pd.DataFrame(Y_test, columns=['original_target'])
dftest['Pred_target']=Y_pred
print( "Predicted values by our model: ", np.round( Y_pred[:3], 2))
print( "Real values                  : ", Y_test[:3] ) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




