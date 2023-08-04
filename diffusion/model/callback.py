import json
import pickle
import numpy as np

from pathlib import Path


class CustomCallback():
    '''
    provides custom callback that stores, prints and saves
    training logs in json-format
    '''
    # settings read from config (set as class attributes)
    args = ['version', 'n_epochs', 'keys_print', 
            'freq_log', 'freq_print']
    
    def __init__(self, config):
        
        # load class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg]) 

        # determines digits for 'fancy' log printing
        self.digits = int(np.log10(self.n_epochs)+1)

        # create log from config file
        self.log = config.copy()

        # create model folder
        self.model_path = Path('logs', self.version)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # create log file path
        self.log_file = self.model_path.joinpath('log.json')
        self.weights_file = self.model_path.joinpath('weights.pkl')
        

    def write_logs(self, logs, epoch):
        '''
        is called during network training
        stores/prints training logs
        '''
        # store training logs
        if (epoch % self.freq_log) == 0:
            # exception errors catch different data formats
            for key, item in logs.items():
                # append if list already exists
                try:
                    self.log[key].append(item.numpy().astype(np.float64))
                # create list otherwise
                except KeyError:
                    try:
                        self.log[key] = [item.numpy().astype(np.float64)]
                    # if list is given
                    except AttributeError:
                        self.log[key] = item

        # print training logs
        if (epoch % self.freq_print) == 0:

            s = f"{epoch:{self.digits}}/{self.n_epochs}"
            for key in self.keys_print:
                try:
                    s += f" | {key}: {logs[key]:2.2e}"
                except KeyError:
                    pass
            print(s)
            

    def save_logs(self):
        '''
        saves recorded training logs in json-format
        '''
        with open(self.log_file, "w") as f:
            json.dump(self.log, f, indent=2)

        print("*** logs saved ***")
        
        
    def save_weights(self, neural_net):
        
        with open(self.weights_file, 'wb') as pickle_file:
            pickle.dump(neural_net.get_weights(), pickle_file)
            
        print("*** weights saved ***")
