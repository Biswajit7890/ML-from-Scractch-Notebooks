#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from scipy.stats import mode 
from sklearn.neighbors import KNeighborsClassifier 


# In[12]:


class K_Nearest_Neighbors_Classifier():
    def __init__( self, K ):
        self.K = K 
    def fit( self, X_train, Y_train ):
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.m, self.n = X_train.shape
    def predict( self, X_test ):
        self.X_test = X_test 
        self.m_test, self.n = X_test.shape
        Y_predict = np.zeros( self.m_test )
        for i in range( self.m_test ):
            x = self.X_test[i]
            neighbors = np.zeros(self.K)
            neighbors = self.find_neighbors(x)
            Y_predict[i] = mode( neighbors )[0][0]
        return Y_predict    
    def find_neighbors(self, x):
        euclidean_distances = np.zeros( self.m )
        for i in range(self.m):
            d = self.euclidean(x,self.X_train[i])
            euclidean_distances[i]=d
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K]     
    def euclidean(self, x, x_train):
        return np.sqrt( np.sum( np.square( x - x_train ) ) ) 
        
            


# In[14]:


df = pd.read_csv( "C:/Users/user/Desktop/IVY WORK BOOK/MACHINE LEARNING/Python Datasets/Classification Datasets/train.csv" )


# In[41]:


df.head(20)


# In[34]:


Gender_map={'Male':0,'Female':1}
df['Gender']=df['Gender'].map(Gender_map)
Married_map={'Yes':1,'No':0}
df['Married']=df['Married'].map(Married_map)
Edu_map={'Graduate':1,'Not Graduate':0}
df['Education']=df['Education'].map(Edu_map)
sel_map={'No':0,'Yes':1}
df['Self_Employed']=df['Self_Employed'].map(sel_map)
prp_map={'Urban':0,'Semiurban':1,'Rural':2}
df['Property_Area']=df['Property_Area'].map(prp_map)
Loan_map={'Y':1,'N':0}
df['Loan_Status']=df['Loan_Status'].map(Loan_map)


# In[42]:


#df=df.drop(labels='Loan_ID', axis=1)
#df=df.drop(labels='LoanAmount', axis=1)
df=df.drop(labels='Dependents', axis=1)


# In[39]:


df.isnull().sum()


# In[22]:


df['Gender']=df['Gender'].bfill()
df['Married']=df['Married'].ffill()
df['Dependents']=df['Dependents'].ffill()
df['Self_Employed']=df['Self_Employed'].bfill()
df['Loan_Amount_Term']=df['Loan_Amount_Term'].bfill()
df['Credit_History']=df['Credit_History'].ffill()


# In[24]:


df.nunique()


# In[47]:


df.shape


# In[ ]:





# In[53]:


X = df.iloc[:,:-1].values 
Y = df.iloc[:,-1:].values
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/3, random_state = 0) 
model = K_Nearest_Neighbors_Classifier( K =80)
model.fit( X_train, Y_train ) 
Y_pred = model.predict(X_test)
correctly_classified =0
count = 0
for count in range(np.size(Y_pred)):
    if Y_test[count]==Y_pred[count]:
        correctly_classified = correctly_classified + 1
print( "Accuracy on test set by our model	 : ", (correctly_classified / count ) * 100 )         
        
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




