# 123

import sklearn.neural_network
import numpy
import matplotlib.pyplot

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
#     predictions = NN.predict(Ts)
    score = NN.score(Ts, Hs)
    return score

def learning_curve(Ts, Hs, filename):
    # INSERT YOUR CODE HERE 

    return max_iters, scores

def self_checker(*args): 
    X, y = learning_curve(*args)
    print (type(X), X)
    print (type(y), y)
    import hashlib
    print (hashlib.md5(open(args[2], "rb").read()).hexdigest())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    self_checker(numpy.array([1,2]), numpy.array([3,4]), "test.png")
    print()
    self_checker(numpy.array([1,2,3,4]), numpy.array([-1,-1,-1,-1]), "test.pdf")
