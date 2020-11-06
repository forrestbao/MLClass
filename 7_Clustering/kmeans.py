# BSD 3-Clause 
# Forrest Sheng Bao
# Copyright 2020

import matplotlib.pyplot as plt 
import numpy, numpy.random, numpy.linalg
import matplotlib.animation as animation
import sklearn.datasets
import copy

def gen_data():
  centers = [(1, 1), (1.2, 1.2)]
  X, _ = sklearn.datasets.make_blobs(n_samples=200, centers=centers, shuffle=False)
  return  X

def closest(x, C):
  """Given a list of centroids in C, return the index of the centroid closest to a sample x
  """
  distances = numpy.linalg.norm(C-x, axis=1) 
  return numpy.argmin(distances)

def kmeans(X, k, C):
  """Run k-means given centroids C 

  X: 2-D numpy array, with samples in rows and features in columns. 
  S: list of integers, each of which is an index to a sample
  """

  # S = [[]]*k  # weird why it doesn't work 
  S = [[] for _ in range(k)]
  # print ("centroid before", C)

  # assignment step 
  for i in range(len(X)):
    x = X[i]
    # print ("sample", i, x)
    cluster_index = closest(x, C)
    # print ("assign to cluster", cluster_index)
    S[cluster_index].append(i)
  
  # print ("S", S)  
  # print ("cluster 0 coordiante", X[S[0]])
  # print ("cluster 1 coordiante", X[S[1]])

  # update centroids 
  # new_C = copy.deepcopy(C)
  for i in range(k):
    C[i] = numpy.sum(X[S[i]], axis=0)/len(S[i])

  # print ("C",)

  return C, S

def plot_k(X, C, S):
  """Given C and S, visualize 
  """
  color_map = {0:'blue', 1:'red'}
  for i in range(len(C)):
    # print ("cluster", i)
    this_cluster = X[S[i]] #2D numpy array
    plt.plot(this_cluster[:,0], this_cluster[:,1], '.', c=color_map[i])
    plt.plot(C[i][0], C[i][1], "P", markersize=12, c=color_map[i])
  
def animate(frame, X, k, C):
  print (C)

  C, S = kmeans(X, k, C)

  fig.clear()
  plot_k(X, C, S)

  # _ = input("enter to continue")

k = 2
X = gen_data()

# Initialize k centroids  by randomly picking k
C = X[numpy.random.choice(X.shape[0], k, replace=False)]
# C = numpy.array([[1.1, 1.1], [1.2, 1.2]])

fig = plt.figure()
ani = animation.FuncAnimation(fig, animate, frames=10, fargs=(X,k, C))
# plt.show()
FPS = 1

ani.save("kmeans.gif", writer='imagemagick', fps=FPS)
# ani.save("kmeans.mp4", writer = animation.FFMpegWriter(fps=FPS) )