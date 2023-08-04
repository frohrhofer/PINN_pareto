import tensorflow as tf


class Loss():
    '''
    provides the physics loss function class
    '''
    # settings read from config (set as class attributes)
    args = ['Re']
    
    def __init__(self, pinn, config):
        
        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg]) 

        # save neural network (weights are updated during training)
        self.pinn = pinn
        
        
    def BC(self, X_BC, U_BC):
        '''
        Standard MSE Loss for boundary conditions
        '''          
        U_pred = self.pinn(X_BC)
        loss_BC = tf.reduce_mean(tf.square(U_BC[:,0:2] - U_pred[:, 0:2]))
        return loss_BC
    
    
    def F(self, X):
        '''
        Physics loss based on Navier-Stokes equation
        '''             
        # tape forward propergation to retrieve gradients
        with tf.GradientTape() as t:
            t.watch(X)
            with tf.GradientTape() as tt:
                tt.watch(X)
                U = self.pinn(X)
            U_d = tt.batch_jacobian(U, X)        
        U_dd = t.batch_jacobian(U_d, X)
        
        # U shape: (x_col, f_i)
        u, v = U[:, 0], U[:, 1]
        
        # U_d shape: (x_col, f_i, dx_i)
        u_x, u_y = U_d[:, 0, 0], U_d[:, 0, 1]
        v_x, v_y = U_d[:, 1, 0], U_d[:, 1, 1] 
        p_x, p_y = U_d[:, 2, 0], U_d[:, 2, 1] 
        
        # U_dd shape: (x_col, f_i, dx_i, dx_j)
        u_xx, u_yy = U_dd[:, 0, 0, 0], U_dd[:, 0, 1, 1]
        v_xx, v_yy = U_dd[:, 1, 0, 0], U_dd[:, 1, 1, 1]
        
        # Navier-Stokes
        res_x = u*u_x + v*u_y + p_x - (u_xx + u_yy) / self.Re
        res_y = u*v_x + v*v_y + p_y - (v_xx + v_yy) / self.Re
        
        # determine residual errors 
        loss_F_x = tf.reduce_mean(tf.square(res_x))
        loss_F_y = tf.reduce_mean(tf.square(res_y))
    
        return loss_F_x, loss_F_y
    
              
