# Project idea 1: Organic chemistry 

Organic chemistry is hard. So let's ML tackle it. 

Project report due: As late as Dec 7. Early turn-in before Thanksgiving, Nov. 26 will give you 10% bonus. 

# Synopsis 

Dr. Yinjie Tang and his team at Washington University of St Louis are studying a bacterium. 
The bacteria could convert a mixture three gases, namely, [CO](https://en.wikipedia.org/wiki/Carbon_monoxide), [CO2](https://en.wikipedia.org/wiki/Carbon_dioxide), and [H2](https://en.wikipedia.org/wiki/Hydrogen), into five kinds of organics, namely, [acetate](https://en.wikipedia.org/wiki/Acetate), [biomass](https://en.wikipedia.org/wiki/Biomass), [butanol](https://en.wikipedia.org/wiki/Butanol), [butyrate](https://en.wikipedia.org/wiki/Butyrate) and [ethanol](https://en.wikipedia.org/wiki/Ethanol). 
The conversion process is a very complicated biochemical kinetic process that biologists and chemists are not quite sure. So, let's build a machine learning model to make the prediction. 

# Data

The experiment was carried out by giving the baterium with different mixtures of the three gases, for example, 20% CO, 50% CO2, and then 30% H2, at different rates, say 10 cubic milliliter per second. Each combination of the three gases and the rate defines a **case**. A case could be experimented for multiple **trials**. 

The five kinds of organics produced by the bacterium from the three gases are called **yields**. They are measured for multiple times throughout the duration of a trial. Usually the yields increase as a trial progresses. 

The data is given in [this CSV file](organic_chemistry_data.csv). The column names are very self-explantory. The first two columns are case ID and trial ID. The third column is the time points that measurements of the yields happened. X1 to X4 are the four variables that defines a case. Y1 to Y6 are the five kinds of yields. To protect the data before publishing, the meaning of X1 to X4 and Y1 to Y6 will not be given. X1 to X4 for rows of the same case ID are the same -- meaning that the gas mixture ratio and the rate are contants in a case. The data has been transformed in several non-inversible steps. 

We will need to build a regression model that can predict Y1 to Y6 given X1 to X4 and time. Hence, the input vector should have 5 (or more when modeling data as time series, see bonus task) elements (X1 to X4, plus time) while the output vector should have 5 elments as well (Y1 to Y4). 

# Tasks
We will use the data in different ways to practice common ML workflows. 
1. [30%] Use all case data together. Do 10-fold cross validation and Leave-one-out cross validation. In this task, you can ignore the case and trial ID. 
2. [70%] Leave-one-case-out cross validation. In each fold or round of cross validation, use data from all cases but one as training set and data from the one left out as test. Report the results on different test sets individually. 
3. [Bonus, up to 40%] The measurements in each trial or case could be viewed as a time series obtain thru a temporal course. So you could use the measurements in multiple time points as input to capture the trend or trajectory of the data. 
<!-- 4. [Bonus, up to 40%] Feature engineering. Because of the small amounts of data, the machine learning model can easily overfit. You are encouraged to manually design features  -->

## Types of models
Please try out several approaches to build regressors. You must include at least the follows: 
1. Support Vector Regressors (SVR)
2. Neural networks (also called multilayer perceptron regressor)
3. Random forests 

If you use the same ML framework (e.g., Scikit-learn), then you may just need to slightly modify your code because most ML frameworks keep the methods consistent for different types of models. For example, in scikit-learn, all training is `.fit()`, and it has the same cross validation framework for any model -- just provide different parameter dictionaries. 

For bonus task, try one type of models is enough. 

## Hyperparameters
The performance of a model highly depends on the choice of its hyperparameters. 
Please tune at least the following hyperparameters:
1. For SVR, try RBF kernel and tune both C and $\sigma$. 
2. For NNs, try 3 different numbers of layers, and 2 different number of neurons in each layer, the regularization weight $\alpha$, and maximal number of iterations. 
3. For random forests, try 3 different maximum heights of trees, 3 minimal Gini impurity or information gain to stop splitting nodes. 
4. If you used other types of models, contact the instructor in advance. 

## What to include in your report 
For each type of models, and each task, report
1. Hyperparameters that you tune and the ranges in which you tune them
2. How you tune or select hyperparameters (e.g., cross validation) -- please include both configurations and results 
3. The performance of your models in most optimal hyperparameters. Evaluate your models in terms of at least RMSE, Spearman's correlation coefficient, and Pearson's correlation coefficient. Feel free to include results using other loss or score functions. Use a mixture of tables and visuals to present. **Report performances on both the training and test sets**
4. [Bonus, 20%] Visualization. For each case, and each Y in Y1 to Y6, visualize the Y vs. time points. Plot both the ground truth and the prediction in two lines. 

### Code: 
Turn in your code. It can be one script that does everything, or a notebook that does things step by step, with or without annotations. Comments or docstrings are not required by encouraged. Light comments are good enough. Do not comment in details. 

[Bonus, 10%] A repo set up online showing incremental development of your code. Just provide a link to the repo in your report. 

# Preprocessing 
The data has a huge dynamic range for each input and output, even time. The range and mean vary from column to column. 
As mentioned in class, especially in the discussion about ANNs, it's better to scale different columns into the same range. You want to scale the data along column, e.g., sklearn's MinMaxScaler.  But do not scale the case and trial columns -- you do not use them as inputs nor outputs of your regression model. **Please include your preprocessing configurations in your report** 

