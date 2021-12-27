from os import error
import numpy as np
 

#These are the choices for cost functions
#The first part of class name represents the activation fn for output layer
#and the second part is actually the cost function that is used.

class sigmoid_quadraticCost(object):
    @staticmethod
    def last_layer_activation(x):
        return ANN.sigmoid(x)

    @staticmethod
    def cost_f(ans, output): #The quadratic cost function
        return 0.5*np.linalg.norm(ans-output)**2

    @staticmethod
    def delta_L(ans, output): 
        return (output-ans)*ANN.sigmoid_derivative(output)


class sigmoid_cross_entropyCost(object):
    @staticmethod
    def last_layer_activation(x):
        return ANN.sigmoid(x)

    @staticmethod
    def cost_f(ans, output): #The cross-entropy cost function
        entropy_fn = -(ans*np.log(output)+(1-ans)*np.log(1-output))
        return np.sqrt(np.linalg.norm(entropy_fn))

    @staticmethod
    def delta_L(ans, output):
        return output-ans


class softmax_log_likelihoodCost(object):
    @staticmethod
    def last_layer_activation(x):
        return ANN.softmax(x)

    @staticmethod
    def cost_f(ans, output): #The log-likelihood cost function
        return -np.nan_to_num(np.log(sum(ans*output)))
    
    @staticmethod
    def delta_L(ans, output):
        return output-ans


class ANN(object):
    """
    The Artificial Neural Network Framework for Deep Learning 
    """
    costfns = {
        "quad": sigmoid_quadraticCost,
        "cross-entropy": sigmoid_cross_entropyCost,
        "log-likelihood": softmax_log_likelihoodCost
    }

    #The functions
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def sigmoid_derivative(y):
        return y*(1-y)

    @staticmethod
    def softmax(x):
        return np.exp(x)/sum(np.exp(x))
    

    def __init__(self, lst, choice_c="quad"):
        #This choice is for cost fn and last layer activation function 
        self.costfn = self.costfns[choice_c]
        #Configuring the weights and biases matrices for each layer
        self.__cfg_wts__bss__(lst)     
        return

    def __cfg_wts__bss__(self, lst):
        """
            Randomly assigning the values to the weights and biases...
        """
        self.weights = list()         #List of weights from each layer
        self.biases = list()         #List of biases from each layer
        
        #Randomly intialising the weights and biases for all layers
        for i in range(len(lst)-1):
            self.weights.append(np.random.randn(lst[i], lst[i+1]))
            #self.weights.append(np.zeros((lst[i], lst[i+1])))
            self.biases.append(np.random.randn(lst[i+1], 1))
            #self.biases.append(np.zeros((lst[i+1], 1)))

        return


    #The main control function of the network
    def main(self, train_data, rate= 0.1, no_epochs= 1, mini_batch_size= 1, test_data= None, 
            test_fn= None, lmbda_l2= 0.0):
        """
            This is the main control function of the network which controls the training and testing
            I'm naming it as main for now.
            Here train_data is the list of the n number of lists which are (input, output) pairs in which
            input is a (784 x 1) matrix and the output is (10, 1) matrix.
        """

        no_data_sets = len(train_data)  #Total no of data sets

        #Iterating the training set no of epochs times
        for epoch_id in range(no_epochs):
            self.C = 0
            np.random.shuffle(train_data)  #Shuffling the data before each epoch

            #Iterating over all the mini batches
            total_batches = int(np.ceil(no_data_sets/mini_batch_size))

            for batch_id in range(total_batches):
                #Call the feedForward function with the mini batch
                _from = mini_batch_size*batch_id
                _to   = _from + mini_batch_size
                batch = train_data[_from:_to]
                dw, db = self.__feedForward__(batch, rate)
                self.__alter__(dw, db, rate, len(batch), lmbda_l2/no_data_sets)

            #Display the progress
            print("\nTraining epoch {0} of {1} completed.".format(epoch_id+1, no_epochs))
            print("Average Cost per training input: ", self.C/no_data_sets)
            if test_data and test_fn:
                self.__test_network__(test_data, test_fn)                
                        
        return
    

    #The Feed Forward function
    def __feedForward__(self, mini_batch, rate= 1, isTest= False):
        """
            (This function works over the whole of mini batch data)

            In feed forward we transfer the input of a layer to another layer after passing 
            it through the activation function. 
            X(j)(l+1) = act_f((W1j(l)*X(1)(l) + W2j(l)*X(2)(l) + ... + ) + B(j)(l)) 
            In general, 
            X(l+1) = act_f(W.T * X(l) + B(l)), T = Transpose 
            
            Here act_f is the activation function 
        """

        activations_list = []  #This is the list storing X values at each layer for all input numbers
            
        for no in range(len(mini_batch)):
            x = mini_batch[no][0]   #The input 
            activations = [x]       #This list stores X values at each layer for this input x

            #Use the formulae to feed forward the data through each layer except last one
            for w,b in zip(self.weights[:-1], self.biases[:-1]):
                assert w.shape[0] == x.shape[0], "Size mismatch at feedForward at layer {0} \nWeight matrix size {1} \nInput matrix size {2}".format(self.weights.index(w)+1, w.T.shape, x.shape)
                x =  self.__activation_f__(np.matmul(w.T, x) + b)
                activations.append(x)
            
            #For last layer output 
            w, b = self.weights[-1], self.biases[-1]
            x =  self.costfn.last_layer_activation(np.matmul(w.T, x) + b)
            activations.append(x)
            activations_list.append(activations)
            
        if isTest:
            #If this is a test then mini batch size will be 1 and we will return the output
            return activations_list[0][-1]

        return self.__backPropagation__(mini_batch, activations_list, rate)     

    
    #The activation function and its derivative function
    def __activation_f__(self, x, function= "sigmoid"):
        """
            The activation function converts the number to a number in the range 0 to 1
            using an activating fn (sigmoid function in this case)
        """
        if (function == "sigmoid"):
            return 1/(1+np.exp(-x)) #Network.sigmoid(x)
        else:
            pass
        
    def __activation_f_derivative__(self, y, function= "sigmoid"):
        """ 
            Here y is the output of the activation function. So y = act_f(x), for any x
        """
        if (function == "sigmoid"):
            return y*(1-y) #Network.sigmoid_derivative(y)
        else:
            pass

   
    #The backpropagation function
    def __backPropagation__(self, mini_batch, activations_list, rate):
        """
            This function calculates the delta for the last layer using the cost function
            and then backpropagates the error to calculate the delta W and delta bfor each 
            layer. It does this for each element in the batch.
        """
        dw_list = [np.zeros(w.shape) for w in self.weights]   #The error list for all weights 
        db_list = [np.zeros(b.shape) for b in self.biases]    #The error list for all biases
        
        for x, activations in zip(mini_batch, activations_list):
            #x is the input data and activations is its activation list
            ans = x[1]                  #The real world answer
            output = activations[-1]    #The solution acc. to network

            #The total average error amount
            self.C += self.costfn.cost_f(ans, output)

            #Derivative of cost fn wrt pre-activated solution vector
            delta = self.costfn.delta_L(ans, output)

            #Back-propagating the error
            for j in range(len(self.weights)-1, -1, -1):
                dw_list[j] += activations[j]*delta.T
                db_list[j] += delta
                delta = np.matmul(self.weights[j], delta)*self.__activation_f_derivative__(activations[j])

        return dw_list, db_list


    #Function to tweek the weights and biases
    def __alter__(self, dw, db, rate, n, reg_factor):
        #reg_factor is L2 regularisation factor (lmbda) / no of training inputs
        for i in range(len(self.weights)):
            self.weights[i] -= reg_factor*self.weights[i]  #Apply L2 regularisation
            self.weights[i] = self.weights[i] - rate*dw[i]/n
            self.biases[i] = self.biases[i] - rate*db[i]/n
        return


    #Function to test the network after each epoch
    def __test_network__(self, test_data, test_function):
        """
            Test fx should take 2 inputs: The output of network and the actual answer and
            should return True, if output is correct and False otherwise.
        """
        counter = 0
        try:
            for item in test_data:
                counter += int(test_function(self.__feedForward__([item], isTest= True), item[1]))
        except Exception as err:
            print("Invalid test Function!!")
        print("{0}/{1}".format(counter, len(test_data)))
        return
        
    """
	    The end 
    """
    