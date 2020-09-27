import numpy as np 
import matplotlib.pyplot as plt

import numpy # import again 
import matplotlib.pyplot # import again 

import numpy.linalg 
import numpy.random


def generate_data(Para1, Para2, seed=0):
    """Generate binary random data

    Para1, Para2: dict, {str:float} for each class, 
      keys are mx (center on x axis), my (center on y axis), 
               ux (sigma on x axis), ux (sigma on y axis), 
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.

    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']), 
                       numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']), 
                       numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack(( Para1['y']*numpy.ones(Para1['N']), 
                       Para2['y']*numpy.ones(Para2['N'])  ))            
    X = numpy.hstack((X1, X2)) 
    X = numpy.transpose(X)
    return X, Y 

def plot_mse(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> # print (X, y)
    >>> plot_mse(X, y, 'test.png')
    array([-1.8650779 , -0.03934209,  2.91707992])
    """
    w = np.array([0,0,0]) # just a placeholder

    # your code here
    
    # limit the range of plot to the dataset only
    matplotlib.pyplot.xlim(numpy.min(X[:,0]), numpy.max(X[:,0]))
    matplotlib.pyplot.ylim(numpy.min(X[:,1]), numpy.max(X[:,1]))
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close('all') # it is important to always clear the plot
    return w

def plot_fisher(X, y, filename): 
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_fisher(X, y, 'test.png')
    array([-1.61707972, -0.0341108 ,  2.54419773])
    >>> X,y = generate_data(\
        {'mx':1.5,'my':2, 'ux':0.1, 'uy':2, 'y':1, 'N':200}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=1)
    >>> plot_fisher(X, y, 'test.png')
    array([-0.2243741 , -0.00264881,  0.40329499])
    """

    w = np.array([0,0,0]) # just a placeholder

    # your code here 

    # limit the range of plot to the dataset only
    matplotlib.pyplot.xlim(numpy.min(X[:,0]), numpy.max(X[:,0]))
    matplotlib.pyplot.ylim(numpy.min(X[:,1]), numpy.max(X[:,1]))
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close('all') # it is important to always clear the plot
    return w


if __name__ == "__main__":
    import doctest
    doctest.testmod()