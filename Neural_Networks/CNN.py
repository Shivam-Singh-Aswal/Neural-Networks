import numpy as np


class ConvoLayer(object):
    def __init__(self, receptive_field_size_list):
        self.__cfg_wts__bss__(receptive_field_size_list)
    
    def __cfg_wts__bss__(self, rF_size_list, layers_list):
        """
            Randomly assigning the values to the weights and biases...
            rF_size_list is receptive field size list
        """
        self.weights = list()         #List of weights from each layer
        self.biases = list()         #List of biases from each layer
        
        #Randomly intialising the weights and biases for all layers
        for i in range(len(layers_list)+1):
            self.weights.append([np.random.randn(rF_size_list[i], rF_size_list[i]) for j in range(layers_list[i])])
            self.biases.append([np.random.random() for j in range(layers_list[i])])

        return


class FullyConnectedLayer(object):
    def __init__(self):
        return


class CNN(object):
    """
    The Convolutional Neural Network Framework for deep Learning 
    """
    def __init__(self, layers_list, receptive_field_size_list):
        
        self.no_layers = []
        return
    
    
    def main():
        return
    
    def __feedForward__(self, mini_batch):
        activations = []
        x = None
        for layer in range(1, len(self.layers)):
            y = np.zeros(self.layers[layer], self.layers[layer])
            self.weights
            w, b = self.weights[layer-1], self.biases[layer-1]
            for i in range(self.layers[layer]):
                for j in range(self.layers[layer]):
                    y[i][j] = sum(w*x[i, i+]) + b 
            x = y
            activations.append(x) 