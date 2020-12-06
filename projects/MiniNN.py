#< ---- 20 char ---->< ---- 20 char --->< ---- 20 char --->< ---- 20 char --->
# 
#    MiniNN is a simple neural network library crafted by Forrest Sheng Bao 
# at Iowa State University for students to hack to understand NNs. 
#    MiniNN is deveoped because the source code of scikit-learn.neural_network 
# or Flux was too long to study and not easy to hack. Not to mention the
# complexity of source code of Tensorflow or PyTorch for teaching intro to ML.
#    With MiniNN, students can easily see gradients on all weights between 
#  layers during training and visualize every layer, and tweak around.

#    Under 200 lines, it covers all core operations of NNs: 
# feedforward, backpropagation, and gradient descent. 
#    To be mini and simple, it supports only logistic/sigmoid activation,
# cross entropy loss, and updating the NN with one sample each time. 
# Preinitialized transfer/weight matrixes are needed to intialize the network. 

# Feel free to use it in your teaching or playing. 
# Licensed under BSD 3-clause 
# Copyright 2020 Forrest Sheng Bao
# Contact him for any suggestions or questions: forrest dot bao aT Gmail 
# Opinions expressed here do not reflect those of Iowa State University

import numpy 
import numpy.random
from sklearn.utils import gen_batches
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=3, floatmode="fixed")

class MiniNN: 
  """

  Naming convention: Any self variable starting with a capitalized letter and ending with s is a list of 1-D or 2-D numpy arrays, each element of which is about one layer, such as weights from one layer to its next layer. 

  self.Ws: list of 2-D numpy arrays, tranfer matrixes of all layers, ordered in feedforward sequence 
  self.phi: activation function 
  self.psi: derivative of activation function, in terms of its OUTPUT, ONLY when phi is logistic 
  self.Xs: list of 2-D numpy arrays, output from each layer
  self.Deltas: list of 2-D numpy arrays, delta from each layer

  """
  def logistic(self, x):
    return 1/(1 + numpy.exp(-x)) 

  def logistic_psi(self, x):
    """If the output of a logistic function is x, then the derivative of x over 
    the input is x * (1-x)
    """
    return x * (1-x)

  def __init__(self, hidden_layer=[5], L2=False, lambd=0.01, batch_size=1, seed=1):
    """Initialize an NN

    hidden_layer: does not include bias
    """
    # replace Ws by hidden_layer
    self.hidden_layer = hidden_layer  # hidden layers
    self.L = len(hidden_layer) + 1  # number of transfer matrix
    self.phi = self.logistic  # same activation function for all neurons
    self.psi = self.logistic_psi
    self.L2 = L2  # L2 regularization
    self.lambd = lambd  # hyperparameter of L2 regularization
    self.batch = batch_size  # size of batch
    self.seed = seed  # seed of random matrix

  def feedforward(self, x, W, phi):
      """feedforward from previou layer output x to next layer via W and Phi
      return an augmented out where the first element is 1, the bias 

      Note the augmented 1 is redundant when the forwarded layer is output. 

      x: 1-D numpy array, augmented input
      W: 2-D numpy array, transfer matrix
      phi: a function name, activation function
      """

      return  numpy.concatenate(([1], # augment the bias 1
              phi(
                    numpy.matmul( W.transpose(), x )  
                ) # end of phi
              )) # end of concatenate

  def predict(self, X_0):
    """make prediction, and log the output of all neurons for backpropagation later 

    X_0: 1-D numpy array, the input vector, AUGMENTED
    """
    Xs = [X_0]; X=X_0
    # print (self.Ws)
    for W in self.Ws:
      # print (W,X, self.phi)
      X = self.feedforward(X, W, self.phi)
      Xs.append(X)
    self.Xs = Xs
    self.oracle = X[1:] # it is safe because Python preserves variables used in for-loops

  def backpropagate(self, delta_next, W_now, psi, x_now):
    """make on step of backpropagation 

    delta_next: delta at the next layer, INCLUDING that on bias term  
                (next means layer index increase by 1; 
                 backpropagation is from next layer to current/now layer)
    W_now: transfer matrix from current layer to next layer (e.g., from layer l to layer l+1)
    psi: derivative of activation function in terms of the activation, not the input of activation function
    x_now: output of current layer 
    """
    delta_next = delta_next[1:] # drop the derivative of error on bias term 

    # first propagate error to the output of previou layer
    delta_now = numpy.matmul(W_now, delta_next) # transfer backward
    # then propagate thru the activation function at previous layer 
    delta_now *= self.psi(x_now) 
    # hadamard product This ONLY works when activation function is logistic
    return delta_now

  def get_deltas(self, target):
    """Produce deltas at every layer 

    target: 1-D numpy array, the target of a sample 
    delta : 1-D numpy array, delta at current layer
    """
    delta = self.oracle - target # delta at output layer is prediction minus target 
                                 # only when activation function is logistic 
    delta = numpy.concatenate(([0], delta)) # artificially prepend the delta on bias to match that in non-output layers. 
    self.Deltas = [delta] # log delta's at all layers

    for l in range(len(self.Ws)-1, -1, -1): # propagate error backwardly 
      # technically, no need to loop to l=0 the input layer. But we do it anyway
      # l is the layer index 
      W, X = self.Ws[l], self.Xs[l]
      delta = self.backpropagate(delta, W, self.psi, X)
      self.Deltas.insert(0, delta) # prepend, because BACK-propagate

  def print_progress(self):
    """print Xs, Deltas, and gradients after a sample is feedforwarded and backpropagated 
    """
    print ("\n prediction: ", self.oracle)
    for l in range(len(self.Ws)+1): 
      print ("layer", l)
      print ("        X:", self.Xs[l], "^T")
      print ("    delta:", self.Deltas[l], "^T")
      if l < len(self.Ws): # last layer has not transfer matrix
        print ('        W:', numpy.array2string(self.Ws[l], prefix='        W: '))
      try: # because in first feedforward round, no gradient computed yet
           # also, last layer has no gradient
        print(' gradient:', numpy.array2string(self.Grads[l], prefix=' gradient: '))
      except: 
        pass
      
  def update_weights(self):
    """ Given a sequence of Deltas and a sequence of Xs, compute the gradient of error on each transform matrix and update it using gradient descent 

    Note that the first element on each delta is on the bias term. It should not be involved in computing the gradient on any weight because the bias term is not connected with previous layer. 
    """
    self.Grads = []
    for l in range(len(Ws)): # l is layer index
      x = self.Xs[l]
      delta = self.Deltas[l+1]
      # print (l, x, delta)
      gradient = numpy.outer(x, delta[1:])
      self.Ws[l] -= 1 * gradient  # descent! 

      self.Grads.append(gradient)
    
    # show that the new prediction will be better to help debug
    # self.predict(self.Xs[0])
    # print ("new prediction:", self.oracle)

  def train(self, x, y, max_iter=100):
    """feedforward, backpropagation, and update weights
    The train function updates an NN using one sample. 
    Unlike scikit-learn or Tensorflow's fit(), x and y here are not a bunch of samples. 

    Homework: Turn this into a loop that we use a batch of samples to update the NN. 

    x: 1-D numpy array, an input vector
    y: 1-D numpy array, the target

    """

    # determine the input layer size and output layer size accroding to x and y
    if len(x.shape) == 1:
      lenx = len(x) - 1
      leny = len(y)
      sample_count = 1
    else:
      lenx = x.shape[1] - 1
      leny = y.shape[1]
      sample_count = len(x)

    self.sample_count = sample_count
    self.hidden_layer.append(leny)
    self.hidden_layer.insert(0, lenx)  # insert input layer and output layer

    # initialize transfer matrix, use random matrix to form
    Ws = []
    numpy.random.seed(self.seed)
    for i in range(self.L):
      Wi = numpy.random.random((self.hidden_layer[i] + 1, self.hidden_layer[i + 1]))
      Ws.append(2 * Wi - 1)
    self.Ws = Ws
    # print(Ws)

    for epoch in range(max_iter):   
      print ("epoch", epoch, end=":")
      self.predict(x) # forward 
      print (self.oracle)
      self.get_deltas(y) # backpropagate
      if verbose:
        self.print_progress()   
      self.update_weights() # update weights, and new prediction will be printed each epoch

if __name__ == "__main__":

  # this part use mak_classification to create samples for the test. the sample size 100, 2 classes.
  n = 2
  x, y = make_classification(n_samples=1000, n_features=2, n_informative=n, n_classes=n, n_redundant=0)
  bias = numpy.ones((len(x), 1))
  x_0 = numpy.hstack((bias, x))
  y_0 = numpy.zeros((len(y), n))
  for i in range(len(y)):
    y_0[i][y[i]] = 1

  # slpit samples into train and test
  x_train, x_test, y_train, y_test = train_test_split(x_0, y_0, test_size=0.2, random_state=1)

  hidden = [8]
  MNN = MiniNN(hidden_layer=hidden, L2=False, lambd=0.01, batch_size=1, seed=1)  # initialize an NN

  MNN.train(x_train, y_train, max_iter=10)
  print("train:")
 





# In[ ]:




