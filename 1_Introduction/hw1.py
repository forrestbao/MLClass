# 123

import sklearn.neural_network
import numpy
import matplotlib.pyplot

import hashlib
import itertools 

def test_NN(Ts, Hs, max_iter=200):
    NN = sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=(4,4), 
        activation='tanh', 
        random_state = 1, 
        max_iter=max_iter
        )
    Ts = Ts.reshape(-1, 1) # learned from error
    NN.fit(Ts, Hs)
    predictions = NN.predict(Ts)
    score = NN.score(Ts, Hs)
    return predictions, score

def generate_data(N):
    g = 9.8
    
    # If using Python's own random number generator
#     Ts = [random.random() for _ in range(N)]
#     Hs = [g*(t**2)/2 for t in Ts]

    numpy.random.seed(1)
    # If using Numpy random number generator
    Ts = numpy.random.rand(N)
    Ts.sort()
    Hs = g*(Ts**2)/2
    
    return Ts, Hs

def learning_curve(N, max_iter, filename="test.png"):
    """

    N: int, number of data points
    max_iter: int, number of iterations to train the neural network
    filename: str, name of the file to save the plot
    """
    
    # INSERT YOUR CODE BELOW


    # INSERT YOUR CODE ABOVE 

    matplotlib.pyplot.savefig(filename) # save the plot to a file 

    # The return line is for self-checking purposes
    return hashlib.md5(open(filename, "rb").read()).hexdigest()

def f(a, b, c):
    """
    a, b: 1-D numpy.ndarray
    c: str, placeholder
    """
    return a+b, a-b, a*b

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print (
        learning_curve(100, 200, "test.png")
    )
