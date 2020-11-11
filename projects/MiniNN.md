# Project Idea 2: MiniNN 

For **due date**, see [the readme file](./). 

[MiniNN](../6_Neural_Networks/MiniNN.py) is a simple NN implementation. There are various tasks you could do around it. 

The tasks below are independent. You can do any one, or any combination of them. Some tasks are small and some tasks are big. Finishing a small task will give a small bonus. For example, if a project's point is 50% of project part, you will get 10 points (50% times 20 out of 100) in final grade that can bump your letter grade one level up. Some tasks are so big, so project point can be more than 100%. There is no cap on the project points that you can earn. You can even do multiple projects and have points add up. 

1. [40% of project part] Currently, MiniNN requires the user to provide pre-initialized transfer matrixes `Ws`. Modify it such that the user just need to provide the number of non-bias neurons in each layer, like `hidden_layer=(100,100,20)` in scikit-learn's `neural_network` module. Note that the dimension of input or output could vary from problem to problem. So you code needs to detect the dimension of training data and adapt. 
2. [150% of project part] Modify MiniNN if necessary, and train a network that can classify images in MNIST dataset. Turn in a report of the performance of your network. Show results using different hyperparameters. For MNIST dataset, you can load it directly using the data loader provided by an existing ML framework, e.g., `sklearn.datasets.load_digits` or `tf.keras.datasets.mnist.load_data`. 
3. [25% of project part] Add L2 regularization to MiniNN and allow users to opt in and out. Default is no L2 regularization. 
4. [60% of project part] Currently, MiniNN updates transfer matrixes after every training sample. As mentioned in class, gradient descent is every expensive. Modify the MiniNN code to enable batch update and allow uses to select batch size. Default 1.  


## How to submit: 
Create a fork of our class repo. Then work in your branch. When done, make a pull request and the instructor will grade. If your editing is more than 15 lines, you must show incremental commits instead of one big commit at the end. 
