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

  def __init__(self, Ws=None, SampleList=None):
    """Initialize an NN

    hidden_layer: does not include bias 
    """
    self.samples = SampleList
    self.Ws = Ws
    self.AverageGradients = []
    self.setAveGradientsZero(self.Ws)
    self.L = len(Ws) # number of layers 
    self.phi = self.logistic # same activation function for all neurons
    self.psi = self.logistic_psi

  def setAveGradientsZero(self, Ws):
  	self.AverageGradients = []
  	for W in Ws:
  		self.AverageGradients.append(numpy.zeros(W.shape))

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

  def predict(self, sample):
    """make prediction, and log the output of all neurons for backpropagation later 

    X_0: 1-D numpy array, the input vector, AUGMENTED
    """
    X = sample.getX()
    sample.addValueLayer(X)
    # print (self.Ws)
    for W in self.Ws:
      # print (W,X, self.phi)
      X = self.feedforward(X, W, self.phi)
      sample.addValueLayer(X)

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

    # first propagate error to the output of previous layer
    delta_now = numpy.matmul(W_now, delta_next) # transfer backward
    # then propagate thru the activation function at previous layer 
    delta_now *= self.psi(x_now) 
    # hadamard product This ONLY works when activation function is logistic
    return delta_now

  def get_deltas(self, sample):
    """Produce deltas at every layer 

    target: 1-D numpy array, the target of a sample 
    delta : 1-D numpy array, delta at current layer
    """
    sample.clearDeltas()
    delta = numpy.subtract(sample.getOutputPrediction()[1:], sample.getY())  # delta at output layer is prediction minus target 
                                 									 # only when activation function is logistic 
    delta = numpy.concatenate(([0], delta)) # artificially prepend the bias on delta to match that in non-output layers. 
    sample.getDeltas().insert(0, delta)

    for l in range(len(self.Ws)-1, -1, -1): # propagate error backwardly 
      # technically, no need to loop to l=0 the input layer. But we do it anyway
      # l is the layer index 
      W, X = self.Ws[l], sample.getLayer(l)
      delta = self.backpropagate(delta, W, self.psi, X)
      sample.getDeltas().insert(0, delta) # prepend, because BACK-propagate
      
  def update_AverageGradientWeights(self, sample):
    """ Given a sequence of Deltas and a sequence of Xs, compute the gradient of error on each transform matrix and update it using gradient descent 

    Note that the first element on each delta is on the bias term. It should not be involved in computing the gradient on any weight because the bias term is not connected with previous layer. 
    """
    
    for l in range(len(self.AverageGradients)): # l is layer index
      x = sample.getLayer(l)
      delta = sample.getDeltas()[l + 1]
      # print (l, x, delta)
      gradient = numpy.outer(x, delta[1:])
      self.AverageGradients[l] = numpy.add(self.AverageGradients[l], gradient)
    
    # show that the new prediction will be better to help debug
    # self.predict(self.Xs[0])
    # print ("new prediction:", self.oracle)

  def averageSumOfGradients(self, size):
  	for i in range(len(self.AverageGradients)):
  		self.AverageGradients[i] = numpy.true_divide(self.AverageGradients[i], size)

  def update_weights(self):
  	for l in range(len(Ws)):
  		self.Ws[l] -= 1 * self.AverageGradients[l]

  def train(self,max_iter=100):
    """feedforward, backpropagation, and update weights
    The train function updates an NN using one sample. 
    Unlike scikit-learn or Tensorflow's fit(), x and y here are not a bunch of samples. 

    Homework: Turn this into a loop that we use a batch of samples to update the NN. 

    x: 1-D numpy array, an input vector
    y: 1-D numpy array, the target

    """
    for epoch in range(max_iter):   
      print ("epoch", epoch, end=":")
      self.setAveGradientsZero(self.Ws)

      for s in self.samples:
      	s.clearLayers()
      	self.predict(s) # forward 

      	self.get_deltas(s)
      	
      	self.update_AverageGradientWeights(s)
      

      self.averageSumOfGradients(len(self.samples))	
      self.update_weights()
      # self.get_deltas(y) # backpropagate
      # self.update_weights() # update weights, and new prediction will be printed each epoch


class Sample:
	def __init__(self, x, y):
		self.x = numpy.array(x)
		self.y = numpy.array(y)
		self.currentValues = []
		self.currentValues.append(numpy.array(x))
		self.deltas = []

	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def addValueLayer(self, x):
		self.currentValues.append(numpy.array(x))

	def clearLayers(self):
		self.currentValues = []

	def getLayer(self, index):
		return self.currentValues[index]

	def getOutputPrediction(self):
		return self.currentValues[len(self.currentValues) - 1]

	def getDeltas(self):
		return self.deltas

	def clearDeltas(self):
		self.deltas = []


if __name__ == "__main__": 


  print("Reading in data...")
  # Read in all data from the file
  inArray = numpy.genfromtxt('train.csv',delimiter=',')
  # Get rid of first row, this row is not needed
  inArray = numpy.delete(inArray, obj = 0, axis = 0)
  print("Done reading data...")

  samps = []

  #Like the example, let user select this, set first and last values to be length of x - 1 and length of y.
  nonBiasTerms = [inArray.shape[1] - 1,15,15,10]

  # Go through each row in the input array and split each row into its label and x values
  for row in inArray:
  	y = numpy.array([0,0,0,0,0,0,0,0,0,0])
  	# set the value at the index of the label to be 1, if the sample is a 0, set index 0 to be a 1
  	y[int(row[0])] = 1
  	# The x values are located from the first index in the row to the end, length of x right now is 784
  	xInputs = row[1:len(row)]
  	# Add the bias term to the beginning of the sample
  	xInputs = numpy.insert(xInputs, 0,1.)

  	samps.append(Sample(xInputs, y))

  
  # Counter to keep track of current element within the nonBiasTerms list
  count = 0
  # Initialize the array of Ws
  Ws = []
  for x in nonBiasTerms[:-1]:
  	#Append a W for each layer in the list except the last layer, assume one bias term
  	Ws.append(numpy.random.rand(x + 1, nonBiasTerms[count + 1]))
  	# Increse counter to properly index into the next term in the list
  	count += 1


  MNN = MiniNN(Ws=Ws, SampleList = samps) # initialize an NN with the transfer matrixes given, as well as the samples to train the NN
  print("Training...")
  MNN.train(max_iter = 10)







