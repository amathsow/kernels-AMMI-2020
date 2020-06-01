#!/usr/bin/env python
# coding: utf-8

# ## Kernel Methods AMMI 2020¶
# 
# #### This is a data challenge for the course "Kernel Methods" for AMMI 2020
# 

# In[42]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


# ### Data loading and processing

# In[43]:


X_train=pd.read_csv('Xtr.csv', sep=',')
Y_train=pd.read_csv('Ytr.csv', sep=',')
X_test=pd.read_csv('Xte.csv', sep=',')


# In[44]:


X_train.shape , Y_train.shape , X_test.shape


# In[45]:


X_train.tail()


# In[46]:


Y_train.tail()


# In[47]:


X_test.tail()


# ### Function to convert a DNA sequence string to a numpy array

# In[48]:


import numpy as np
import re
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))


# In[49]:


my_string ='AACAGTZZTA'
string_to_array(my_string)


# ### Function to encode a DNA sequence string as an ordinal vector
# Returns a numpy vector with `a=0.25, c=0.50, g=0.75, t=1.00, n=0.00`

# In[50]:


def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # T
    float_encoded[float_encoded == 4] = 0.00 # anything else, z
    return float_encoded


# So let's try it out with a simple short sequence:

# In[51]:


test_sequence = 'AACGCGCTTNN'
ordinal_encoder(string_to_array(test_sequence))


# ### One-hot encoding DNA sequence data

# In[52]:


# function to one-hot encode a DNA sequence string
# non 'acgt' bases (n) are 0000
# returns a L x 4 numpy array
from sklearn.preprocessing import OneHotEncoder



def one_hot_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded


# So let's try it out with a simple short sequence:

# ### Here is a function that can be used to convert any sequence (string) to overlapping k-mer words:

# In[53]:


def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


# In[54]:


def spectrum_bow(data,k=5):
    data = data
    dim = 101- k+1
    out = []
    header = ['word'+str(i) for i in range(dim)]
    for indx in range(len(data)):
        line = data.iloc[indx]['seq']
        line  = getKmers(line, k)
        out.append(line)
        #print(line)
    out = np.array(out)
    out = pd.DataFrame(data=out,columns=header)
    return out


# In[55]:


modified_x_train = spectrum_bow(X_train,7)
modified_x_test = spectrum_bow(X_test,7)


# In[56]:


from sklearn.preprocessing import OneHotEncoder

def oneHot_enc(data):
    onehot = OneHotEncoder(sparse=False)
    data = onehot.fit_transform(data)
    return data


# In[57]:


train_test_conc = pd.concat([modified_x_train,modified_x_test])


# In[58]:


train_test_conc_onehot = oneHot_enc(train_test_conc)


# In[59]:



# modified_x_train_onHot = oneHot_enc(modified_x_train)
# modified_x_test_onHot = oneHot_enc(modified_x_test)


# In[60]:


modified_x_train_onHot, modified_x_test_onHot = train_test_conc_onehot[:2000,:], train_test_conc_onehot[2000:,:]


# In[61]:


modified_x_train_onHot


# In[62]:


modified_x_test_onHot


# ### kernel implementations
# 

# In[63]:


def rbf_kernel(X1, X2, sigma=10):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the RBF kernel with parameter sigma
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    sigma: float
    '''
    # For loop with rbf_kernel_element works but is slow in python
    # Use matrix operations!
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K

def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)


# In[64]:


def linear_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the linear kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return X1.dot(X2.T)

def quadratic_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the quadratic kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return (1 + linear_kernel(X1, X2))**2  


# In[65]:


class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'quadratic': quadratic_kernel,
        'rbf': rbf_kernel,
    }
    def __init__(self, kernel='quadratic', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = kwargs.get('sigma', None)
        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass


# In[66]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[67]:


class KernelRidge():
    '''
    Kernel Ridge Regression
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, sigma=None, lambd=0.1):
        self.kernel = rbf_kernel
        self.sigma = sigma
        self.lambd = lambd

    def fit(self, X, y):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        
        # Compute default sigma from data
        if self.sigma is None:
            self.sigma = sigma_from_median(X)
        
        A = self.kernel(X, X, sigma=self.sigma) + n * self.lambd * np.eye(n)
        
        ## self.alpha = (K + n lambda I)^-1 y
        # Solution to A x = y
        self.alpha = np.linalg.solve(A , y)

        return self
        
    def predict(self, X):
        # Prediction rule: 
        K_x = self.kernel(X, self.X_train, sigma=self.sigma)
        return K_x.dot(self.alpha)


# In[68]:


class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.01, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y, sample_weights=None):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        if sample_weights is not None:
            w_sqrt = np.sqrt(sample_weights)
            self.X_train = self.X_train * w_sqrt[:, None]
            self.y_train = self.y_train * w_sqrt
        
        A = self.kernel_function_(X, X, **self.kernel_parameters) # + n * self.lambd * np.eye(n)
        
        A[np.diag_indices_from(A)] += n*self.lambd
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train,**self.kernel_parameters )
        return  np.sign(K_x.dot(self.alpha))
    
    def predict(self, X):
        return self.decision_function(X)
    def evaluate(self,y,yy):
        acc = 0
        for i in range(len(y)):
            if y[i]==yy[i]: acc+=1
        return acc/len(y)


# In[69]:



class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelLogisticRegression, self).__init__(**kwargs)

    def fit(self, X, y, max_iter=100, tol=1e-5):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        K = self.kernel_function_(X, X, **self.kernel_parameters)
        
        # IRLS
        KRR = KernelRidgeRegression(
            lambd=2*self.lambd,
            kernel=self.kernel_name,
            **self.kernel_parameters
        )
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            m = K.dot(alpha_old)
            w = sigmoid(m) * sigmoid(-m)
            z = m + self.y_train / sigmoid(self.y_train * m)
            alpha = KRR.fit(self.X_train, z, sample_weights=w).alpha
            
            
            
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol:
                break

        self.n_iter = n_iter
        self.alpha = alpha

        return self
            
    def decision_function(self, X):
        K_x = self.kernel_function_(X_test, self.X_train, **self.kernel_parameters)
        # Probability of y=1 (between 0 and 1)
        return sigmoid(K_x.dot(self.alpha))

    def predict(self, X):
        probas = self.decision_function(X)
        predicted_classes = np.where(probas < 0.5, -1, 1)
        return predicted_classes


# In[70]:


y = np.array(Y_train.drop(['Id'],axis = 1)['Bound'])


# In[71]:


for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1


# ## cross varidation

# In[ ]:


from sklearn.model_selection import KFold
kfold=KFold(n_splits=8)
l = []
for i, (train_index, validate_index) in enumerate(kfold.split(modified_x_train_onHot)):
        X_train, y_train = modified_x_train_onHot[train_index], y[train_index]
        X_valid, y_valid = modified_x_train_onHot[validate_index], y[validate_index]
        model_ = KernelRidgeRegression(0.0001)
        y_pred = model_.fit(X_train , y_train ).predict(X_valid )
        acc = model_.evaluate(y_pred,y_valid)
        l.append(acc)
        print('Fold ',i+1,' has accuracy of',acc,'\n\n')
s = sum(l)/len(l)
print('average is ',s)       
        


# ## Training and predicting on the whole data

# In[227]:


model = KernelRidgeRegression(0.001)
y_pred = model.fit(modified_x_train_onHot , y ).predict(modified_x_test_onHot ) 


# In[228]:


y_pred.shape


# In[229]:


pred = []
for i in range(len(y_pred)):
    if y_pred[i]==-1:
        pred.append(0)
    else: pred.append(1)


# In[230]:


key = np.array([i for i in range(len(pred))])
pred = np.array(pred)
submsionFile = {'Id':key,'Bound':pred}
submsionFile = pd.DataFrame(submsionFile)


# In[231]:


submsionFile.to_csv('modelPredictions.csv',index=False)


# In[ ]:




