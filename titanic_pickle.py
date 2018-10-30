
# coding: utf-8

# In[ ]:


from autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
import random
from random import randrange
import matplotlib.pyplot as plt; plt.rcdefaults()
import pickle


# In[ ]:


class MyLogisticReg :
    
#A class for Linear Regression Models
    def __init__(self, weight=[], num_iter=50000, method='gd',lamda=0.01,test_method = 'r', test_1_2 = False,alpha=0.001):

        #if not specified, my settings are:
        #iterations: 1000
        #method: gradient descent
        #lambda: 0.01
        #alpha: 0.001
        #weight will be set in the set_weight function
        
        self.initWeight = weight
        self.lamda = lamda
        self.method = method
        self.alpha = alpha
        self.num_iter = num_iter
        self.weight = weight
        self.test_method = test_method
        self.test_1_2 = test_1_2

    #compute sigmoid
    def __sigmoid(self, z):
        
        #replace any z value that is less
        #than -500 with the value -500
        z[z<-500] = -500
        return 1 / (1 + np.exp(-z))
        
    #calculate the loss
    #theta is the weight
    
    def loss_function(self,X, y, theta):
        
        ita = np.dot(X,theta)
        
        #==================================================================================================
        # prevent overflow
        #==================================================================================================
        right = np.sum(y[ita < 100] * ita[ita < 100] - np.log(1+np.exp(ita[ita < 100] ))) +                 np.sum(y[ita >= 100] * ita[ita >= 100] - ita[ita > 100] ) 
        #==================================================================================================
        
        left = (self.lamda/2)*(np.dot(theta,theta))
        answer = left - right
        return answer
    
    #if weight is not specified
    #I will assign random number from 0.01 to 0.1
    def set_weight(self, new_weight, X_dupe):
        
        if len(new_weight) == 0:
            self.weight = np.random.randint(low=1, high=10, size=X_dupe.shape[1])
            self.weight = np.divide(self.weight, 100)
            self.weight = np.array(self.weight)

        else:
            self.weight = self.initWeight
            self.weight = np.array(self.weight)
            
        
    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)
        w_0 = np.ones((len(X),1))
        X_dupe = np.array(X)
        
        #stack a column of one to the original X
        #which corresponds to w0
        X_dupe = np.hstack((w_0,X_dupe))
        
        #set the weight
        self.set_weight(self.weight, X_dupe)
        
        #w_history will store all the past weights
        w_history = np.zeros((self.num_iter, X_dupe.shape[1]))
        d = X_dupe.shape[1]
        
        
        if self.method == 'gd':


            for i in range(self.num_iter):
                #calculate ita
                z = np.dot(X_dupe,self.weight)             
                h = self.__sigmoid(z)
                right = np.dot((y-h),X_dupe)/y.size
                
                #making a dupe weight so that
                #I can set w0 to 0
                #since the derivative of (w.T)(w) is 2w
                #w0 is not part of it
                dupe_weight = self.weight
                dupe_weight[0] = 0
                left = self.lamda * dupe_weight
                gradient = left - right
                
                #store the weight
                w_history[i] = self.weight
                
                #update the weight
                self.weight -= self.alpha * gradient
                
                #plot iterations vs converge curve
                mean = self.loss_function(X_dupe, y, self.weight)
                
                
                #calculate diff to see if it is smaller than termination condition
                #which is smaller than 10**-4
                if i >= 100:
                    time_1 = 1/(d)
                    sum_3 = np.sum(abs(self.weight-w_history[i-100,:]))
                    
                    diff = time_1 * sum_3
                    if diff < (10**-4):
                        break;
        

        if self.method == 'sgd':

            #in this case, our initial w0 will be a 10 by 1 matrix
            #since we only take 10 data each iteration
            w_00 = np.ones((10,1))
            data = np.hstack((np.vstack(y),X))

            
            for i in range(self.num_iter):              
                data = np.random.permutation(data)
                #make a dupe X with an extra column
                X_dupe_train = data[:10,1:]
                y_train = data[:10, 0]
                X_dupe_train = np.hstack((w_00,X_dupe_train))
                #calculate ita
                z = np.dot(X_dupe_train,self.weight)
                h = self.__sigmoid(z)
                    
                right = np.dot((y_train-h),X_dupe_train) / y_train.size
                dupe_weight = self.weight
                dupe_weight[0] = 0
                left = self.lamda * dupe_weight
                gradient = left - right 
                w_history[i] = self.weight

                self.weight -= self.alpha * gradient
                
                #plot iterations vs converge curve
                mean = self.loss_function(X_dupe, y, self.weight)

             
                if i >= 100:
                    time_1 = 1/(d)
                    sum_3 = np.sum(abs(self.weight-w_history[i-100,:]))
                    
                    diff = time_1 * sum_3
                    if diff < (10**-4):
                        break;

        
    #this function is to return the weight
    def return_weight(self, k):

        if len(self.weight)!= 0:
            k = self.weight
        else:
            k = self.initWeight
        return k
    
    #predict the y values
    def predict(self, X):
        X = np.array(X)
        y_pred = np.zeros(X.shape[0])
        self.weight = np.array(self.weight)
        
        X_copy = X
        
        for i in range(0,len(y_pred)):

            #y = ita*w + w0
            each_pre = np.dot(self.weight[1:],X_copy[i]) + self.weight[0]
            if each_pre > 0:
                y_pred[i] = 1
            else:
                y_pred[i] = 0        
        return y_pred
        


# In[ ]:


def evaluate( y_test , y_pred ):

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    error_rate = (np.sum(np.equal(y_test,y_pred).astype(np.float))/ y_test.size)
    return error_rate


# In[ ]:


#this function is to do cross_validation
#according to n_folds
#and will take the weight that corresponds to
#the largest accuracy
#then I will run the entire set
#using this weight
#then to predict the y values
def cross_validation(dataset, n_folds):
    cross_validation_dataset = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset_copy)/n_folds)
    
    for i in range(0, n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        cross_validation_dataset.append(fold)
    
    #cvd would be a n_folds x (len(x))/n_folds x features mat
    cvd = np.array(cross_validation_dataset)
    list_of_weights = []
    max = 0
    
    for i in range(0, n_folds):
        x_test = cvd[i][:,1:]
        y_test = cvd[i][:,0]
        
        x_train = np.array(list())
        y_train = np.array(list())
        
        for j in range(0,n_folds):
            if i == j:
                continue;
            else:
                for q in range(0,len(cvd[i])):

                    
                    if(len(x_train)) == 0:
                        x_train = np.append(x_train,cvd[j,q,1:])
                        y_train = np.append(y_train,cvd[j,q,0])
                    
                    else:
                        x_train = np.vstack(((x_train),cvd[j,q,1:]))
                        y_train = np.hstack(((y_train),cvd[j,q,0]))
        

        model = MyLogisticReg()
        model.fit(x_train,y_train)
        ypred = model.predict(x_test)
        a = evaluate(y_test,ypred)
        
        if a > max:
            max = a
            list_of_weights = model.return_weight(list_of_weights)

    return list_of_weights


# In[ ]:


def main():

    #data is imported and shuffled
    #and divided into X and Y
    data = pd.read_csv('titanic_train.csv')
    data = np.random.permutation(data)
    X = np.array(data[:,1:])
    Y = np.array(data[:, 0])
    
    #titanic set
    final_w= cross_validation(data,10)  
    
    model = MyLogisticReg(final_w)
    pickle.dump(model, open("titanic_classifier.pkl", "wb"))

    
    test_pickle = pickle.load(open("titanic_classifier.pkl", "rb"))
    ypred = test_pickle.predict(X)
    k = evaluate(Y,ypred)
    print(k)
    
    


# In[ ]:


main()


# In[ ]:


#0.8014084507042254

