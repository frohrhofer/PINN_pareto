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
    '''
    # settings read from config (set as class attributes)
    args = ['seed', 'n_hidden', 'n_neurons', 'activation',
            'feature_scaling', 'kappa', 'L', 'lambda_tau',
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
        self.x_min, self.x_max = 0, self.L        
        self.t_min, self.t_max = 0, self.lambda_tau * self.L**2 / self.kappa

        self.x_range = self.x_max - self.x_min
        self.t_range = self.t_max - self.t_min
        self.X_min = [self.x_min, self.t_min]
        self.X_max = [self.x_max, self.t_max]
        self.X_range = [self.x_range, self.t_range]

        print('*** PINN build & initialized ***')
        

    def build_network(self, verbose):
        '''
        builds the basic PINN architecture based on
        a Keras 'Sequential' model
        '''
        # create keras Sequential model
        self.neural_net = Sequential()
        # build input layer
        self.neural_net.add(InputLayer(input_shape=(2,)))
        # build hidden layers
        for i in range(self.n_hidden):
            self.neural_net.add(Dense(units=self.n_neurons,
                                      activation=self.activation))
        # build linear output layer
        self.neural_net.add(Dense(units=1, activation=None))
        if verbose:
            self.neural_net.summary()
            
            
    def scale_features(self, X):
        '''
        MinMax Feature Scaling to range [0, 1]
        '''      
        X_scaled = (X - self.X_min) / self.X_range
        
        return X_scaled
            

    def call(self, X):
        '''
        Overrite call of the (outer) PINN network to use feature scaling
        '''
        if self.feature_scaling:
            X = self.scale_features(X)
        return self.neural_net(X)

        

    def train(self):
        '''
        trains the PINN
        '''
        # learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate) 
                                   
        # Adam optimizer with default settings for momentum
        self.optimizer = Adam(learning_rate=lr_schedule)    
        
        print("Training started...")   
        for epoch in range(self.n_epochs):
            
            # sample and extract training data
            datasets = self.data_loader.sample_datasets()   
            X_BC, u_BC = datasets['BC'] 
            X_IC, u_IC = datasets['IC']
            X_col, _ = datasets['col']
            X_test, u_test = datasets['test']

            # perform one train step
            train_logs = self.train_step(X_IC, u_IC, X_BC, u_BC, X_col)
            test_logs = self.test_step(X_test, u_test)
            # combine train and test logs
            logs = {**train_logs, **test_logs}
            # provide logs to callback
            self.callback.write_logs(logs, epoch) 

        # save logs and model weights
        self.callback.save_weights(self.neural_net)
        self.callback.save_logs()
        print("Training finished!")
        

    @tf.function
    def train_step(self, X_IC, u_IC, X_BC, u_BC, X_col):
        '''
        performs a single SGD training step
        '''
        # open a GradientTape to record forward/loss pass
        with tf.GradientTape() as tape:

            # Data loss
            loss_IC = self.loss.u(X_IC, u_IC)
            loss_BC = self.loss.u(X_BC, u_BC)
            loss_u = loss_IC + loss_BC
            
            # Physics loss
            loss_F = self.loss.F(X_col)
          
            # weighted MO LOss
            loss_train = self.alpha * loss_u + (1 - self.alpha) * loss_F

        # retrieve gradients
        grads = tape.gradient(loss_train, self.neural_net.weights)
        # perform single GD step
        self.optimizer.apply_gradients(zip(grads, self.neural_net.weights))

        # save logs for recording
        train_logs = {'loss_train': loss_train, 'loss_u': loss_u, 'loss_F': loss_F}
        return train_logs
    

    def test_step(self, X_test, u_test):
        '''
        Test set performance measures: MSE and rel. L2
        '''
        
        u_pred = self(X_test)
        
        # MSE loss
        loss_test = tf.reduce_mean(tf.square(u_pred - u_test))
        # relative L2 norm       
        L2_test = tf.norm(u_test-u_pred) / tf.norm(u_test)
                                           
        # save logs for recording
        test_logs = {'loss_test': loss_test, 'L2_test': L2_test}   
        return test_logs