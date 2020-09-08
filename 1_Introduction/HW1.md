# Homework 1

Problem 1 [5pts]: 

Suppose there are 4 independent variables. And 3 operations, i.e., addition, multiplication, and power, are allowed in connecting them into an expression. How many different expression can we have? 
For example, let the 4 variabels be A, B , C, and D. Then A+B *  C ^D is one expression while A^B * C+D is another. The order of variables matters, e.g., A+B and B+A are two different expressions. 
Each operator or variable appears in the expression EAXCTLY ONCE. No parentheses. 

The purpose of this problem is to understand how many different parametric equations can exist among a set of variables, and thus using traditional scientific discovery way to model the relationship between high-dimensional variables is very challenging.

Problem 2 [5pts (2.5pts for correct returns, and 2.5pts for correct plot)]:

In the Jupyter Notebook example for Unit 1, there is a demo that the score of a neural network changes along with the maximal number of iterations (i.e., the `max_iter` argument in the function `test_NN`). Now, let's visualize the change by writing a plot function `learning_curve` of the following template: 

```python
def learning_curve(Ts, Hs, filename):
        
    return max_iters, scores
```
where
- the first two arguments `Ts` and `Hs` (1-D numpy array each, e.g., [1,2,3] not [[1],[2], [3]]) are the input and corresponding output for a supervised learning task, 
- the last argument `filename` (string) specifies the filename (Matplotlib uses the suffix to automatically determine the file format) to save the plot,  
- the 1st return `max_iters` (1-D numpy array) is a sequence of maximal numbers of iterations from 50 to 2000 with a step of 50, 
- and the 2nd return `scores` (1-D numpy array) is a sequence of scores returned from the function `test_NN`, each of which corresponds to an element in `max_iters`. 

Make a line plot between the maximal number of iterations and the score of the NN, and save it as `filename`. Just do basic plot (`matplotlib.pyplot.plt(max_iters, scores)`) with all default settings. No labels nor title needed. Do not adjust figure size, resolution, tick values and locations, etc. 


Submit one `.py` file with the answer to Problem 1 as a comment line at the top of the file. Then include the function definition to `learning_curve` below. Do NOT include other code in your submission. 

Pay close attention to the data types. A wrong type will cause the grading script to throw an error and you won't get points. 

Example: 
After exeucting the code below (assuming that `numpy` and `matplotlib.pyplot` are imported as is), 
```
print (learning_curve(numpy.array([1,2]), numpy.array([3,4]), "test.whatever_suffix"))

import hashlib
print (hashlib.md5(open("test.whatever_suffix", "rb").read()).hexdigest())
```
you should see (ignore identation and typesetting)

```
(array([  50,  100,  150,  200,  250,  300,  350,  400,  450,  500,  550,
        600,  650,  700,  750,  800,  850,  900,  950, 1000, 1050, 1100,
       1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650,
       1700, 1750, 1800, 1850, 1900, 1950]), 
array([-24.51278849, -14.0049323 ,  -7.28061896,  -3.24700787,
        -1.02356304,   0.10299754,   0.62719032,   0.85084125,
         0.93854938,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779,   0.95801779,
         0.95801779,   0.95801779,   0.95801779]))
70452c164b57501f1af55d8ab7722cdb
```

where the 1st return is a 1-D numpy array from 50 to 2000 with a step 50, the 2nd is the other 1-D numpy array of corresponding scores, and the 3rd is the MD5 sum of the generated PNG file. 