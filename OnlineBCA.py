"""
Title: OnlineBCA.py

Two Layer Recurrent Neural Network for Bounded Component Analysis

Reference: B. Simsek and A. T. Erdogan, "Online Bounded Component Analysis: A Simple Recurrent Neural Network with Local Update Rule for Unsupervised Separation of Dependent and Independent Sources," 2019

Code Writer: Barışcan Bozkurt (Koç University - EEE & Mathematics)

Date: 17.02.2021
"""


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class OnlineBCA:
    """
    Implementation of online two layer Recurrent Neural Network with Local Update Rule for Unsupervised Seperation of Sources.
    Reference: B. Simsek and A. T. Erdogan, "Online Bounded Component Analysis: A Simple Recurrent Neural Network with Local Update Rule for Unsupervised Separation of Dependent and Independent Sources," 2019
    
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    F              -- Feedforward Synaptic Connection Matrix, must be size of (s_dim, x_dim)
    B              -- Recurrent Synaptic Connection Matrix, must be size of (s_dim, s_dim)
    lambda_        -- Forgetting factor (close to 1, but less than 1)
    
    gamma_hat
    beta
    mu_F
    mu_y
    
    
    Methods:
    ==================================
    
    whiten_signal(X)        -- Whiten the given batch signal X
    
    ProjectOntoLInfty(X)   -- Project the given vector X onto L_infinity norm ball
    
    fit_next(x_online)     -- Updates the network parameters for one data point x_online
    
    fit_batch(X_batch)     -- Updates the network parameters for given batch data X_batch (but in online manner)
    
    """
    
    def __init__(self, s_dim, x_dim, lambda_ = 0.999, mu_F = 0.03, beta = 5, neural_dynamic_iterations = 100, neural_dynamic_tol = 1e-8, F = None, B = None):
        if F is not None:
            assert F.shape == (s_dim, x_dim), "The shape of the initial guess F must be (s_dim, x_dim) = (%d,%d)" % (s_dim, x_dim)
            F = F
        else:
            F = np.random.randn(s_dim,x_dim)
            F = (F / np.sqrt(np.sum(np.abs(F)**2,axis = 1)).reshape(s_dim,1))
            
        if B is not None:
            assert B.shape == (s_dim,s_dim), "The shape of the initial guess B must be (s_dim, s_dim) = (%d,%d)" % (s_dim,s_dim)
            B = B
        else:
            B = np.eye(s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.lambda_ = lambda_
        self.beta = beta
        self.mu_F = mu_F
        self.gamma_hat = (1-lambda_)/lambda_
        self.F = F
        self.B = B
        self.neural_dynamic_iterations = neural_dynamic_iterations
        self.neural_dynamic_tol = neural_dynamic_tol
        
    def whiten_signal(self, X, mean_normalize = True, type_ = 3):
        """
        Input : X  ---> Input signal to be whitened
        type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
        Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
        """
        if mean_normalize:
            X = X - np.mean(X,axis = 0, keepdims = True)

        cov = np.cov(X.T)

        if type_ == 3: # Whitening using singular value decomposition
            U,S,V = np.linalg.svd(cov)
            d = np.diag(1.0 / np.sqrt(S))
            W_pre = np.dot(U, np.dot(d, U.T))

        else: # Whitening using eigenvalue decomposition
            d,S = np.linalg.eigh(cov)
            D = np.diag(d)

            D_sqrt = np.sqrt(D * (D>0))

            if type_ == 1: # Type defines how you want W_pre matrix to be
                W_pre = np.linalg.pinv(S@D_sqrt)
            elif type_ == 2:
                W_pre = np.linalg.pinv(S@D_sqrt@S.T)

        X_white = (W_pre @ X.T).T

        return X_white, W_pre
    
    def ProjectOntoLInfty(self, X):
        
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def fit_next(self,x_current):
        
        lambda_, beta, mu_F, gamma_hat, F, B = self.lambda_, self.beta, self.mu_F, self.gamma_hat, self.F, self.B
        neural_dynamic_iterations = self.neural_dynamic_iterations
        neural_dynamic_tol = self.neural_dynamic_tol
        
        y = np.zeros(self.s_dim)
        
        for i in range(neural_dynamic_iterations):
            mu_y = 0.9 / (i+1)

            e = F @ x_current - y
            
            y_old = y
            
            y = self.ProjectOntoLInfty(y + mu_y*(gamma_hat * B @ y + beta * e))
            
            if np.linalg.norm(y - y_old) < neural_dynamic_tol:
                break
                
        e = F @ x_current - y

        F = F - mu_F * beta * np.outer(e, x_current)

        B = (1/lambda_) * (B - gamma_hat * np.outer(B @ y, B @ y))        
        
        self.F = F
        self.B = B
        
        return y
        
        
    def fit_batch(self, X, n_epochs = 2, whiten = True, whiten_type = 2, shuffle = False, verbose = True):
        
        lambda_, beta, mu_F, gamma_hat, F, B = self.lambda_, self.beta, self.mu_F, self.gamma_hat, self.F, self.B
        neural_dynamic_iterations = self.neural_dynamic_iterations
        neural_dynamic_tol = self.neural_dynamic_tol
        
        assert X.shape[1] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[0]
        
        Y = np.zeros((samples, self.s_dim))
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X, type_ = whiten_type)
        else:
            X_white = X 
            
            
        for k in range(n_epochs):
            if verbose:
                for i_sample in tqdm(range(samples)):
                    x_current = X_white[idx[i_sample], :]
                    y = np.zeros(self.s_dim)

                    for i in range(neural_dynamic_iterations):
                        mu_y = 0.9 / (i+1)

                        e = F @ x_current - y
                        
                        y_old = y
                        
                        y = self.ProjectOntoLInfty(y + mu_y*(gamma_hat * B @ y + beta * e))
                        
                        if np.linalg.norm(y-y_old) < neural_dynamic_tol:
                            break
                            
                    e = F @ x_current - y

                    F = F - mu_F * beta * np.outer(e, x_current)

                    B = (1/lambda_) * (B - gamma_hat * np.outer(B @ y, B @ y))

                    # Record the seperated signal
                    Y[idx[i_sample],:] = y

            else:
                for i_sample in (range(samples)):
                    x_current = X[idx[i_sample], :]
                    y = np.zeros(self.s_dim)

                    for i in range(neural_dynamic_iterations):
                        mu_y = 0.9 / (i+1)

                        e = F @ x_current - y
                        y_old = y
                        
                        y = self.ProjectOntoLInfty(y + mu_y*(gamma_hat * B @ y + beta * e))
                        
                        if np.linalg.norm(y - y_old) < neural_dynamic_tol:
                            break
                            
                    e = F @ x_current - y

                    F = F - mu_F * beta * np.outer(e, x_current)

                    B = (1/lambda_) * (B - gamma_hat * np.outer(B @ y, B @ y))

                    # Record the seperated signal
                    Y[idx[i_sample],:] = y
        self.F = F
        self.B = B
        
        return Y

def whiten_signal(X, mean_normalize = True, type_ = 3):
    """
    Input : X  ---> Input signal to be whitened
    
    type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
    
    Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
    """
    if mean_normalize:
        X = X - np.mean(X,axis = 0, keepdims = True)
    
    cov = np.cov(X.T)
    
    if type_ == 3: # Whitening using singular value decomposition
        U,S,V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        W_pre = np.dot(U, np.dot(d, U.T))
        
    else: # Whitening using eigenvalue decomposition
        d,S = np.linalg.eigh(cov)
        D = np.diag(d)

        D_sqrt = np.sqrt(D * (D>0))

        if type_ == 1: # Type defines how you want W_pre matrix to be
            W_pre = np.linalg.pinv(S@D_sqrt)
        elif type_ == 2:
            W_pre = np.linalg.pinv(S@D_sqrt@S.T)
    
    X_white = (W_pre @ X.T).T
    
    return X_white, W_pre

def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:,i] = ZeroOneNormalizeData(X[:,i])

    return X_normalized

def Subplot_gray_images(I, image_shape = [512,512], height = 15, width = 15):
    
    n_images = I.shape[1]
    fig, ax = plt.subplots(1,n_images)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:,i].reshape(image_shape[0],image_shape[1]), cmap = 'gray')
    plt.show()