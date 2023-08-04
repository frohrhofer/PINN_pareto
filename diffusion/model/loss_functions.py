import tensorflow as tf


class Loss():
    '''
    provides the physics loss function class
    '''
    # settings read from config (set as class attributes)
    args = ['kappa']
    
    def __init__(self, pinn, config):
        
        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg]) 

        # store neural network (weights are updated during training)
        self.pinn = pinn        
        
        
    def u(self, X, u_true):
        '''
        Standard MSE Loss for initial and boundary conditions
        '''
        
        u_pred = self.pinn(X)
        loss_u = tf.reduce_mean(tf.square(u_pred - u_true))
    
        return loss_u
    
    
    def F(self, X):
        '''
        Physics loss based on diffusion equation
        '''
              
        # tape forward propergation to retrieve gradients
        with tf.GradientTape() as t:
            t.watch(X)
            with tf.GradientTape() as tt:
                tt.watch(X)
                u = self.pinn(X)
            u_d = tt.batch_jacobian(u, X)        
        u_dd = t.batch_jacobian(u_d, X)
        
        # U_d shape: (x_col, f_i, dx_i)
        u_t = u_d[:, 0, 1]       
        # U_dd shape: (x_col, f_i, dx_i, dx_j)
        u_xx = u_dd[:, 0, 0, 0]     
        # heat equation
        res_F = u_t - self.kappa * u_xx
        
        loss_F = tf.reduce_mean(tf.square(res_F))
        
        return loss_F
    
              
