import numpy as np
import idx2numpy
import sys

#Importing the Artificial Neural Network
sys.path.insert(0, "./")
from Neural_Networks import ANN


#Loading data from the files
#Defining the function to convert (28, 28) matrix to an array of 784 elements
def load(image_file, labelfile, start, end):
    input_list = idx2numpy.convert_from_file(image_file)[start:end]
    soln_list = list(idx2numpy.convert_from_file(labelfile))[start:end]
    matrix_list = list()
    soln_vectors = list()

    for i in range(end-start):
        #The soln vector
        vector = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])   #Defining basic soln vector
        vector[0][soln_list[i]] = 1
        soln_vectors.append(vector.T)   #Make the soln vector a column vector
        
        #The input data
        data_point = input_list[i]/255
        this_vector = [] #The vector for current number
        for i in data_point:
            this_vector.extend(i)  #add every row of matrix to the array 
        matrix_list.append(np.array([this_vector]).T) #Convert the row vector to column vector 

    return list(zip(matrix_list, soln_vectors))


#Testing function for the network
def testing_function(output, ans):
    return np.argmax(output) == np.argmax(ans) and max(output) >=0.0


def main():
    abs_path_mnistset = "D:/Field\Machine Learning/Digit Recognition/Neural_Networks_and_projects/MNIST_dataset"
    #Loading Training data
    start, end = 0, 10000
    imagefile = abs_path_mnistset + "/train-images.idx3-ubyte"
    labelfile = abs_path_mnistset + "/train-labels.idx1-ubyte"
    train_data = load(imagefile, labelfile, start, end)

    #Loading test data
    start, end = 0, 10000
    imagefile = abs_path_mnistset + "/t10k-images.idx3-ubyte"
    labelfile = abs_path_mnistset + "/t10k-labels.idx1-ubyte"
    testing_data = load(imagefile, labelfile, start, end)

    #Parameters of the neural network
    rate = float(input("Rate : "))
    iters = int(input("Iterations: "))
    batch_size = int(input("Mini batch size: "))
    lmbda = float(input("Lambda: "))

    divvu = ANN([784, 30, 30, 10], choice_c="cross-entropy")  #The network

    #Train the network
    divvu.main(train_data, rate, iters, batch_size, test_data= testing_data, 
        test_fn= testing_function, lmbda_l2= lmbda)


if __name__ == "__main__":
    main()
