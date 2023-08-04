import numpy as np
import tensorflow as tf

from numpy.random import random
from pyDOE import lhs

class DataLoader():
    
    # settings read from config (set as class attributes)
    args = ['seed', 'L', 'lambda_tau', 'kappa', 
            'N_IC', 'N_BC', 'N_col', 'N_test']
    
    def __init__(self, config):
        
        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg]) 
        
        # set seed for data sampling
        np.random.seed(self.seed)

        # spatial domain
        self.x_min, self.x_max = 0, self.L
        # diffusive time scale
        self.tau = self.L**2 / self.kappa
        # time domain
        self.t_min, self.t_max = 0, self.lambda_tau * self.tau
       
        # limits and ranges
        self.x_range = self.x_max - self.x_min
        self.t_range = self.t_max - self.t_min
        self.X_min = [self.x_min, self.t_min]
        self.X_max = [self.x_max, self.t_max]
        self.X_range = [self.x_range, self.t_range]   
        

    def array2tensor(self, array, exp_dim=True):
        '''
        Auxilary function to convert numpy-array to tf-tensor
        expands dimensions if necessary
        '''
        if exp_dim:
            array = np.expand_dims(array, axis=1)

        return tf.convert_to_tensor(array, dtype=tf.float32)
    
    
    def analytical_solution(self, X):
        '''
        Returns analytical solution for diffusion equation
        '''      
        x, t = X[:, 0], X[:, 1]
    
        u = np.sin(np.pi*x/self.L) * np.exp(-np.pi**2*t/self.tau)
        return self.array2tensor(u, exp_dim=True)


    def regular_grid(self, N=128):
        '''
        Provides coordinates and solution on a regular grid
        '''
        # Coordinate ticks
        x = np.linspace(self.x_min, self.x_max, N)
        t = np.linspace(self.t_min, self.t_max, N)
        # Meshgrid
        xx, tt = np.meshgrid(x, t)
        X = self.array2tensor(np.vstack((xx.flatten(), tt.flatten())).T, 
                              exp_dim=False)
        # solution
        u = self.array2tensor(self.analytical_solution(X), 
                              exp_dim=False)
        
        return X, u
    
    
    def sample_IC(self, N=128):     
        '''
        Provides random samples of IC with N data points
        IC: u = sin(pi * x / L)
        '''
        # IC coordinates (t=0)
        x_ticks = np.random.rand(N) * self.x_range 
        X_IC = self.array2tensor([[x, 0] for x in x_ticks], 
                                 exp_dim=False)   
        # IC temperature
        u_IC = self.array2tensor([[np.sin(np.pi*x/self.L)] for x in x_ticks], 
                                 exp_dim=False)
        
        return X_IC, u_IC
 

    def sample_BC(self, N=128):
        '''
        Provides random samples of BC with N data points at each boundary (top and bottom)
        BC: u = 0
        '''
        # top boundary coordinates (x=x_max)
        t_ticks = np.random.rand(N) * self.t_range         
        X_top = [[self.x_max, t] for t in t_ticks]
        # bottom boundary coordinates (x=x_min)
        t_ticks = np.random.rand(N) * self.t_range       
        X_bottom = [[self.x_min, t] for t in t_ticks]
        # add both boundaries and prodive BC temperature
        X_BC = self.array2tensor(X_top + X_bottom, exp_dim=False)   
        u_BC = self.array2tensor([[0] * X_BC.shape[0]], exp_dim=False)           
               
        return X_BC, u_BC
    
    
    def sample_domain(self, N=1024):
        '''
        LHS sampling of coordinates inside function domain
        '''
        # coordinates
        X = self.array2tensor(self.X_min + self.X_range*lhs(2, N), 
                              exp_dim=False) 
        # solution
        u = self.array2tensor(self.analytical_solution(X), 
                              exp_dim=False) 
        
        return X, u
   

    def sample_datasets(self):
        '''
        Provides datasets for IC/BC, collocation and test points
        Uses number of data points as specified in the config file
        '''
        X_IC, u_IC = self.sample_IC(N=self.N_IC)
        X_BC, u_BC = self.sample_BC(N=self.N_BC)
        X_col, u_col = self.sample_domain(N=self.N_col)
        X_test, u_test = self.sample_domain(N=self.N_test)
        
        datasets = {'IC': [X_IC, u_IC],
                    'BC': [X_BC, u_BC],
                    'col': [X_col, u_col],
                    'test': [X_test, u_test]}
        
        return datasets
    
    