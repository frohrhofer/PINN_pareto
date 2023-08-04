import numpy as np
import tensorflow as tf

from model.data_loader import DataLoader
from model.loss_functions import Loss
from model.callback import CustomCallback

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam, SGD


class PhysicsInformedNN(Model):
    '''
    provides the basic Physics-Informed Neural Network class
    with hard constraints for initial conditions
    '''
    # settings read from config (set as class attributes)
    args = ['seed', 'n_hidden', 'n_neurons', 'activation',
            'feature_scaling', 'L',
            'n_epochs', 'learning_rate', 'decay_rate',
            'alpha']
    
    def __init__(self, config, verbose=False):

        # call parent constructor & build NN
        super().__init__(name='PINN')
         # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg]) 
            
        # set random seed for weights initialization
        tf.random.set_seed(self.seed)
        
        # builds network architecture
        self.build_network(verbose)
        # create data loader instance
        self.data_loader = DataLoader(config)
        # create loss instance
        self.loss = Loss(self, config)
        # create callback instance
        self.callback = CustomCallback(config)

         # system domain for feature scaling
        self.x_min, self.x_max = -self.L, self.L
        self.y_min, self.y_max = -self.L, self.L
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.X_min = [self.x_min, self.y_min]
        self.X_max = [self.x_max, self.y_max]
        self.X_range = [self.x_range, self.y_range]
        print('*** PINN build & initialized ***')
        

    def build_network(self, verbose):
        '''
        builds the basic PINN architecture based on
        a Keras 'Sequential' model
        '''
        # nested neural network (for hard constraints)
        self.neural_net = Sequential()  
        # build input layer (x,y)
        self.neural_net.add(InputLayer(input_shape=(2,)))
        # build hidden layers
        for i in range(self.n_hidden):
            self.neural_net.add(Dense(units=self.n_neurons, 
                                      activation=self.activation))
        # build 2d linear output layer (Psi, p)
        self.neural_net.add(Dense(units=2, activation=None))
        # print network summary
        if verbose:
            self.neural_net.summary() 
            

    def scale_features(self, X):
        '''
        MinMax Feature Scaling to range [-1, 1]
        '''            
        X_scaled = 2 * (X - self.X_min) / self.X_range - 1
        
        return X_scaled
            
      
    @tf.function
    def call(self, X):
        '''
        Overrite call of the (outer) PINN network to use feature scaling
        and hard constrained continuity equation
        '''
        if self.feature_scaling:
            X = self.scale_features(X)
    
        with tf.GradientTape() as tape:
            tape.watch(X)  
            U = self.neural_net(X)
        U_d = tape.batch_jacobian(U, X)
        # get u and v from (latent) stream function
        u, v = U_d[:, 0, 1], -U_d[:, 0, 0]
        # get pressure from regular prediction
        p = U[:, 1]
        # stack to 3d output
        return tf.stack([u, v, p], axis=1)
       

    def train(self):
        '''
        trains the PINN
        '''
        # learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate) 
                                   
        # Adam optimizer with default settings for momentum
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)    
        
        print("Training started...")   
        for epoch in range(self.n_epochs):
            
            # sample and extract training data
            datasets = self.data_loader.sample_datasets()   
            X_BC, U_BC = datasets['BC'] 
            X_col, _ = datasets['col']
            X_test, U_test = datasets['test']

            # perform one train step
            train_logs = self.train_step(X_BC, U_BC, X_col)
            test_logs = self.test_step(X_test, U_test)
            # combine train and test logs
            logs = {**train_logs, **test_logs}
            # provide logs to callback
            self.callback.write_logs(logs, epoch) 

        # save logs and model weights
        self.callback.save_weights(self.neural_net)
        self.callback.save_logs()
        print("Training finished!")
        

    @tf.function
    def train_step(self, X_BC, U_BC, X_col):
        '''
        performs a single SGD training step
        '''
        # open a GradientTape to record forward/loss pass
        with tf.GradientTape() as tape:

            # Data loss (on Boundary)
            loss_U = self.loss.BC(X_BC, U_BC)
            
            # Physics loss
            loss_F_x, loss_F_y = self.loss.F(X_col)
            loss_F = loss_F_x + loss_F_y
          
            # weighted mean squared error loss
            loss_train = self.alpha * loss_U + (1 - self.alpha) * loss_F

        # retrieve gradients
        grads = tape.gradient(loss_train, self.weights)
        # perform single GD step
        self.optimizer.apply_gradients(zip(grads, self.weights))

        # save logs for recording
        train_logs = {'loss_train': loss_train, 'loss_U': loss_U, 'loss_F': loss_F}
        return train_logs
    

    def test_step(self, X_test, U_test):
        '''
        Test set performance measures: MSE and rel. L2
        '''     
        u_test = U_test[:, 0]
        v_test = U_test[:, 1]
        p_test = U_test[:, 2] - tf.reduce_mean(U_test[:, 2])
        U_test = tf.stack([u_test, v_test, p_test], axis=1)
        
        U_pred = self(X_test)
        u_pred = U_pred[:, 0]
        v_pred = U_pred[:, 1]
        p_pred = U_pred[:, 2] - tf.reduce_mean(U_pred[:, 2])
        U_pred = tf.stack([u_pred, v_pred, p_pred], axis=1)
        
        # MSE loss
        loss_test_u = tf.reduce_mean(tf.square(u_test - u_pred)) 
        loss_test_v = tf.reduce_mean(tf.square(v_test - v_pred))
        loss_test_p = tf.reduce_mean(tf.square(p_test - p_pred))
        loss_test = tf.reduce_mean(tf.square(U_test - U_pred))
        
        # relative L2 norm       
        L2_test_u = tf.norm(u_test-u_pred) / tf.norm(u_test)
        L2_test_v = tf.norm(v_test-v_pred) / tf.norm(v_test)
        L2_test_p = tf.norm(p_test-p_pred) / tf.norm(p_test) 
        L2_test = tf.norm(U_test-U_pred) / tf.norm(U_test)
                                           
        # save logs for recording
        test_logs = {'loss_test': loss_test, 'L2_test': L2_test, 
                     'L2_test_u': L2_test_u, 'L2_test_v': L2_test_v, 'L2_test_p': L2_test_p}   
        return test_logs