---
title: | 
         CS 474/574 Machine Learning \
         6. Neural Networks
author: |
          Prof. Dr. Forrest Sheng Bao \
          Dept. of Computer Science \
          Iowa State University \
          Ames, IA, USA \
date:   \today
header-includes: |
    \usepackage{amssymb,mathtools,blkarray}
        \usefonttheme[onlymath]{serif}
    \usepackage[vlined,algoruled,titlenotnumbered,linesnumbered]{algorithm2e}
    \usepackage{algorithmic}
    \setbeamercolor{math text}{fg=green!50!black}
    \setbeamercolor{normal text in math text}{parent=math text}
    \newcommand*{\vertbar}{\rule[-1ex]{0.5pt}{2.5ex}}
    \newcommand*{\horzbar}{\rule[.5ex]{2.5ex}{0.5pt}}
    \setlength\labelsep   {.5pt}  
    \setbeamersize{text margin left=3mm,text margin right=4mm} 
    \setlength{\leftmargini}{15pt}
    \usepackage{hyperref}
    \hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    }
    \setlength{\abovedisplayskip}{1pt}
    \setlength{\belowdisplayskip}{1pt}
classoption:
- aspectratio=169
---

#

To compile this file, be sure you first compile all `.tex` files under `figs` folder into PDF. 

`one_hidden_layer_step{1,2,3}.pdf` are produced from `one_hidden_layer.tex` by manually commenting different parts. So the step pdfs are in the repo. 

# Why artificial neural networks (ANNs) work

-   Supervised or Reinforcement learning is about function fitting.

-   To get the function, the analytical form sometimes doesn't matter.

-   As long as we can mimic/fit it accurately enough, it's good.

-   An ANN is a magical structure that can mimic any function \[Fig.
    5.3, Bishop book\], if the ANN is "complex" enough. 
    -- Known as "Universal Approximation." 

# One neuron/perceptron
::: {.columns}
:::: {.column width=0.4}
![](figs/one_neuron_2.pdf)
::::
:::: {.column width=0.6}
-   A **neuron** (also called a **perceptron**) connects its inputs and output as follows: 
  \begin{equation}\hat{y}= \phi(\mathbf{w}^T \mathbf{x}) = \phi(w_1x_1 + w2x_2 + w_dx_d + \underbrace{w_{d+1}}_{bias} x_{d+1})\label{eq:one_neuron} \end{equation} 
  where 
    *  $x_{d+1}=1$ is augmented but often omitted, 
    *  $\mathbf{x} = [x_1, x_2, \dots, x_d]$ is the feature vector of a sample or the raw input. Because in NNs, it is often the raw input, let's call it **input vector**. 
    *  $\phi(\cdot)$ is an **activation function**, which could be nonlinear, e.g., step or logistic ($\sigma$).
<!-- \underbrace{x_{d+1}}_{\text{the augmented 1,\\ often omitted.}}$.  -->
::::
:::

-   This is a linear classifier: $\phi(\mathbf{w}^T \mathbf{x})$ where
    $\phi(\cdot)$ can be, e.g., a step or sign function. (If using a continuous
    function for $h(\cdot)$, we get a regressor. In the regressor case, $h$ is often nonlinear.) 


# Notations and terminology
::: {.columns}
:::: {.column width=0.4}
<!-- ![](figs/one_neuron_2.pdf) -->

We will use this style for a neuron. The inside of a neuron is hidden, and the label on the node is its output. The bias is `optional. 

![](figs/one_neuron.pdf)
::::
:::: {.column width=0.6}
-  A neuron or a neural network is often represented using a directed graph. 
-  On this graph, every **node** is a neuron and every edge is called a **connection**. 
-  How many connections do we have here? 
-  d + 1 (1 for bias)
-  The green neurons **feed** their outputs ($x_1, x_2, \dots, x_d$) to the red neuron. 
<!-- -  The red neuron does not feed into any other neuron. We just take its output and use it, e.g., for prediction.  -->
::::
:::


# A perceptron vs. the perceptron algorithm

-   Why is one algorithm seen earlier called  the "perceptron"
    algorithm?

-   Because $\phi(\mathbf{w}^T \mathbf{x})$ is exactly one
    neuron/perceptron in an ANN.

-   Frank Rosenblatt published his perceptron algorithm in 1962 titled
    "Principles of Neurodynamics: Perceptrons and the Theory of Brain
    Mechanisms."

-   Linearly separable cases only! It cannot even do XOR.

-   Therefore, Marvin Minsky jumped to the conclusion that ANNs were
    useless. \[Perceptrons, Marvin Minsky and Seymour Papert, MIT Press,
    1969\]

-   However, Minsky is an AAAI fellow but not a prophet.

-   If we expand a perceptron into layers of perceptrons, we get an **artificial neural network (ANN)** or an **multi-layer perceptron (MLP)**, which is much more powerful, and for sure, can mimic XOR. 

# From a single neuron to a network of neurons I 
::: {.columns}
:::: {.column width=0.35}
![](figs/one_hidden_layer_step1.pdf){width=100%}
::::
:::: {.column width=0.67}
- I/O relation (recall Eq. \ref{eq:one_neuron}) for the red-circled neuron: 
$\phi \left [
    \begin{pmatrix}
    w_{1,1} & w_{2,1} & \cdots & w_{d,1} & b_1 
    \end{pmatrix}
    \begin{pmatrix}
    x_1 \\ x_2 \\ \vdots \\ x_d \\ 1 
    \end{pmatrix}
    \right ] 
    = \begin{pmatrix}
        o_1
    \end{pmatrix}$
- $o_1$ is the output/activation of the neuron. 
- For a matrix of only one element, we usually abbreviate it into a scalar. Hence $(o_1)$ can be rewritten as $o_1$ when no ambiguity raises.

- Math note: Applying a scalar-domain function to a matrix means applying the function to each element of the matrix. For example, if 
$\phi(x) = \begin{cases}
1,& if~~x>0 \\
-1,& else. 
\end{cases}$
then 
$\phi 
\begin{pmatrix}
 1 & -2 \\
 -3 & 4 
\end{pmatrix}
= \begin{pmatrix}
 1 & -1 \\
 -1 & 1 
\end{pmatrix}$
::::
:::

# From a single neuron to a network of neurons II
::: {.columns}
:::: {.column width=0.35}
![](figs/one_hidden_layer_step2.pdf){width=100%}
::::
:::: {.column width=0.67}
- I/O relation (recall Eq. \ref{eq:one_neuron}) for blue-circled neuron: 
$\phi \left [
    \begin{pmatrix}
    w_{1,2} & w_{2,2} & \cdots & w_{d,2} & b_1 
    \end{pmatrix}
    \begin{pmatrix}
    x_1 \\ x_2 \\ \vdots \\ x_d \\ 1 
    \end{pmatrix}
    \right ] 
    = \begin{pmatrix}
        o_2
    \end{pmatrix}$
- $(o_2)$ abbreviated into $o_2$.
- $o_2$ is the output/activation of the neuron. 
::::
:::

# From a single neuron to a network of neurons III
::: {.columns}
:::: {.column width=0.35}
![](figs/one_hidden_layer_step3.pdf){width=100%}
::::
:::: {.column width=0.72}


- Put results above together:
    $\phi \left [
    \begin{pmatrix}
    w_{1,1} & w_{2,1} & \cdots & w_{d,1} & b_1 \\
    w_{1,2} & w_{2,2} & \cdots & w_{d,2} & b_2
    \end{pmatrix}
    \begin{pmatrix}
    x_1 \\ x_2 \\ \vdots \\ x_d \\ 1 
    \end{pmatrix}
    \right ] 
    = \begin{pmatrix}
        o_1\\o_2
    \end{pmatrix}$

- Rewrite into a shorter form: 
$\phi(\mathbb{W}^T\mathbf{x}) = \mathbf{o}$
where 
$\mathbb{W} = \begin{pmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2} \\
\vdots & \vdots \\
b_1 & b_2 
\end{pmatrix}
= \begin{pmatrix}
\vertbar & \vertbar \\
\mathbf{W}_1 & \mathbf{W}_2 \\
\vertbar & \vertbar \\
  \end{pmatrix}$
::::
:::

# From a single neuron to a network of neurons IV
::: {.columns}
:::: {.column width=0.35}
![](figs/one_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.72}
- Finish the last two connections, $o_1 \rightarrow \hat{y}$ and $o_2 \rightarrow \hat{y}$. 
- The I/O relation: 
  $\hat{y}=\phi(\mathbb{V}^T\mathbf{o})$. 
- In this example, $\mathbb{V}$ has only one column. Why? 
- How to expand for more than two neurons in the middle?
::::
:::

# From a single neuron to a network of neurons V

::: {.columns}
:::: {.column width=0.4}
![](figs/larger_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.62}

- Adding a neuron in middle layer means inseting a new column of weights into $\mathbb{W}$. 

- From $\mathbf{x}=[x_1, x_2, \dots, x_d, 1]^T$ to $\mathbf{o}=[o_1, o_2, o_3]^T$:
    $\phi(\mathbb{W}^T\mathbf{x}) = \mathbf{o}$
    where 
    $\mathbb{W} = \begin{pmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    w_{2,1} & w_{2,2} & w_{2,3}\\
    \vdots & \vdots & \vdots  \\
    b_1 & b_2 & b_3
    \end{pmatrix}
    = \begin{pmatrix}
    \vertbar & \vertbar & \vertbar \\
    \mathbf{W}_1 & \mathbf{W}_2 & \mathbf{W}_3 \\
    \vertbar & \vertbar & \vertbar \\
    \end{pmatrix}
    =
    \underbrace{ [\mathbf{W}^T_1, \mathbf{W}^T_2, \mathbf{W}^T_3]^T}_{\text{just for writing convenience}}$

- From $\mathbf{o}$ to $\hat{y}$: 

  $\hat{y}=
  \phi \left [
  \begin{pmatrix}
    v_1 & v_2 & v_3
  \end{pmatrix}
  \begin{pmatrix}
    o_1 \\ o_2 \\ o_3
  \end{pmatrix}
  \right ]
  = 
  \phi(\mathbb{V}^T\mathbf{o})$

- Pop quiz: Write $\mathbb{V}$ into matrix form. 

::::
:::

# Layers
::: {.columns}
:::: {.column width=0.6}
![](figs/layers.pdf){width=100%}
::::
:::: {.column width=0.4}
- Perceptrons can be grouped into **layers**: 
    * All neurons have no input form the **input layer**
    * All neurons do not output into other neurons form the **output layer**
    * **hidden layers**: between the two special layers above.
- There is only one input layer and only one output layer per ANN. 
::::
:::

- For hidden-layer neurons, if they share the same inputs (not necessarily the input of the ANN), then they belong to the same layer. Mathematically, all neurons, whose outputs $\mathbf{o}$ are resulted from the same transform $\mathbf{o} = \phi(\mathbb{W}^T\mathbf{x})$  where $\mathbf{x}$ is the outputs of previous-layer neurons, belong to the same layer. 

- Each layer (including output layer) can have any number of neurons. Minimal is one. 

- An ANN can have any arbitrary number of hidden layers. Even zero. 

# Feedforward I

- The transform $\mathbf{o} = \phi(\mathbb{W}^T\mathbf{x})$ is called **feedforward** where $\mathbf{x}$ is "fed" from previous layer to current layer to produce $\mathbf{o}$. It is a basic algorithm for an ANN to yield outputs. 

- Recursively feedforward, you can create a complex ANN: 
  $\phi\left ( \mathbb{W}^{(l)T} \cdots \phi \left( \mathbb{W}^{(2)T} \phi(\mathbb{W}^{(1)T}\mathbf{x}) \right)  \right )$ where $\mathbb{W}^{(i)}$ is the weights from layer $i$ to layer $i+1$. 

# Feedforward II

An example ($x_0$ is bias. And yes, the right 3 layers have no biases.): 

::: {.columns}
:::: {.column width=0.45}
![](figs/two_hidden_layers.pdf){width=100%}
::::
:::: {.column width=0.6}
Transition between layers: 
$\begin{blockarray}{c}
        \mathbf{x} \\
      \begin{block}{(c)}
       x_0 \\ x_1 \\ x_2 \\ x_3 \\
      \end{block}
 \end{blockarray} 
\xRightarrow{\phi,  \mathbb{W}^{(1)}}
\begin{blockarray}{c}
      \mathbf{h^{(1)}} \\
      \begin{block}{(c)}
         h_{1}^{(1)}\\ h^{(1)}_{2}\\ h^{(1)}_{3} \\ h^{(1)}_{4} \\
      \end{block}
 \end{blockarray} 
\xRightarrow{\phi,  \mathbb{W}^{(2)}}
\begin{blockarray}{c}
      \mathbf{h^{(2)}} \\
      \begin{block}{(c)}
         h_{1}^{(2}\\ h^{(2)}_{2}\\ h^{(2)}_{3} \\ 
      \end{block}
 \end{blockarray} 
\xRightarrow{\phi,  \mathbb{W}^{(3)}}
\begin{blockarray}{c}
      \mathbf{\hat{y}} \\
      \begin{block}{(c)}
         y_1 \\ y_2 \\
      \end{block}
 \end{blockarray}$

End-to-end relation, from $\mathbf{x}$ to $\mathbf{\hat{y}}$: 

$\mathbf{\hat{y}} = 
\begin{pmatrix}
    \hat{y_1} \\ \hat{y_2}
\end{pmatrix}
=
    \phi \Bigg( \mathbb{W}^{(3)T} 
    \overbrace{    
        \phi \bigg( \mathbb{W}^{(2)T} 
            \underbrace{\phi \left ( \mathbb{W}^{(1)T} \mathbf{x} \right )}
                        _{\left [h_{1}^{(1)}, h^{(1)}_{2},h^{(1)}_{3},h^{(1)}_{4}\right ]^T  } 
            \bigg) 
    }
    ^{\left [h_{1}^{(2)}, h^{(2)}_{2},h^{(2)}_{3} \right ]^T}
    \Bigg)$
::::
:::

# ANN for XOR

![](figs/XOR.png){width=50%}

Any logic operation: Fig. 2.9 of [the Neural Network Ebook comes with the `neuralnetwork` package for LaTeX](https://github.com/battlesnake/neural/blob/master/examples/neural-networks-ebook.pdf)

# Taking a break

Why did mathematicians invent matrixes? 

# Quiz 
In order to approximate the relation $h={1\over 2}gt^2$

- How many neurons are needed in the input layer?

- How many neurons are needed in the output layer? 

- If the activation function is logistic function, are hidden layers necessary? Why? 

- How many hidden neurons are needed at least? 

# Training ANNs
- The power of an ANN is in its weights, e.g., $\mathbf{W}^{(1)}$, $\mathbf{W}^{(2)}, \dots$. 

- By "training" an ANN, we mean tuning its weights. 

- But how? 

- Due to the complexity of ANNs, it's difficult to get an analytical form of the solution of weights like what we did in simple linear classifiers. 

- Gradient descent is commonly used, e.g., $\mathbf{W}^{(i)} \leftarrow \mathbf{W}^{(i)} + \rho {\nabla \text{Loss (such as error)}}$

- But first, we need a loss function. 

# Loss function for training ANNs

- In early times of ANN research, mean squared error (MSE) was used as the loss function:
  ${1\over N}\sum_i (\hat{y_i} - y_i)^2$
  where $y_i$ and $\hat{y}_i$ are ground truth target and prediction for the $i$-th sample, respectively, $N$ is the number of samples. (Note it's difference to sum of squared error in the averaging part )

- Then it is noted that negative logistic loss is better (neg-log-loss) in that the logistic function panelizes a big error more than a small error. (See [an explanation by Shuyu Luo in Towards Data Science](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11)  where $h_\theta (x)$ is the prediction $\hat{y}$ used in our class.)

- We talked about log-loss in Unit 5 Regression. [Google also has a good refreshing material](https://developers.google.com/machine-learning/crash-course/logistic-regression/model-training)

- Given a prediction $\hat{y}$ and a ground truth target $y$, the neg-log-loss is $- y\log \hat{y} - (1-y)(1-\log \hat{y})$

- **The discussion above about neg-log-loss is for classification only. For regression, MSE is still used de facto.**

# Gradient descent in ANNs: simplest case

::: {.columns}
:::: {.column width=0.2}
![](figs/one_neuron.pdf){width=100%}
::::
:::: {.column width=0.8}

- If just one neuron, it's something similar to perceptron algorithm [See the first part of Unit 4 SVMs slides.], except that we will use neg-log-loss instead of (mean or sum) squared error. 

- Warm-up: $\hat{y} = \phi(\mathbf{w}^T\mathbf{x})$

- Gradient for each weight $w_i$ based on chain rule: 
    \begin{equation}
    { \partial E \over \partial w_i} 
        = {\partial E \over \partial \hat{y}} \pause
        {\partial \hat{y} \over \partial \mathbf{w}^T\mathbf{x}} \pause
        {\partial \mathbf{w}^T\mathbf{x} \over \partial w_i} \label{eq:gradient_simplest}
    \end{equation}
- Derivative of log-loss: 
${\partial E \over \partial \hat{y}} = {\partial~ - y\log \hat{y} - (1-y)(1-\log \hat{y}) \over \pause \partial \hat{y} } = {\hat{y} - y \over \hat{y}(1-\hat{y}) } \pause$
- Derivative of activation function (from calculus, the derivative of $\sigma(x) \pause = 1/(1+e^{-x})$ is $\sigma(x)(1-\sigma(x))$): 
${\partial \hat{y} \over \partial \mathbf{w}^T\mathbf{x}}  
={\partial \phi\left (\mathbf{w}^T\mathbf{x} \right ) \over \partial \mathbf{w}^T\mathbf{x}}
= \hat{y}(1-\hat{y})\pause$

- Lastly, ${\partial \mathbf{w}^T\mathbf{x} \over \partial w_i } = x_i \pause$. 

- Put together: Gradient for each weight $w_i$: 
    ${ \partial E \over \partial w_i} = (\hat{y}-y) x_i \pause$ 

- Or in matrix form for all weights: 
    ${ \partial E \over \partial \mathbf{w}} = (\hat{y}-y) \mathbf{x}$. 
::::
:::

# What about neurons more upstream? 
::: {.columns}
:::: {.column width=0.3}
![](figs/one_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.73}
- Things become trickier for a neuron that is not directly connected to the output layer. 

- Use the steps we did in previous slide: $${ \partial E \over \partial w_{1,1}} 
        ={ \partial E \over \partial o_1}
        {\partial \mathbf{o} \over \partial w_{1,1}}$$
    where ${\partial E \over \partial o_1}$ traces error to the neuron destinated by $w_{1,1}$. 

- What is the error over $o_1$? 

- It involves an algorithm called **backpropagation**. 

- Let's see the intuition. 
::::
:::


# The intuition behind backpropagation
::: {.columns}
:::: {.column width=0.5}
![](figs/backprop_intuition.pdf){width=100%}
::::
:::: {.column width=0.5}
- An auto company is losing 1 Million dollars this year. 
- The company has two divisions, producing pickups and sedans, and 30% and 70% of total sales, respectively. 
- Which department is to be blamed more? 
- What're the losses contributed by Pickup division and sedan division respectively? 
::::
:::


# Backpropagation: an example I 
::: {.columns}
:::: {.column width=0.3}
![](figs/one_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.73}
- Let's consider the gradient of error/loss over one weight $w_{1,1}$:
  $${ \partial E \over \partial w_{1,1}} 
        ={ \partial E \over \partial o_1}
        {\partial \mathbf{o} \over \partial w_{1,1}}$$

- Expand the first term (partially making use of results in Eq. \ref{eq:gradient_simplest})
  <!-- \begin{align} -->
  $$
  { \partial E \over \partial o_1} 
  =  {\partial E \over \partial \hat{y} }  
    {\partial \hat{y} \over \partial \mathbf{v}^T \mathbf{o}}  
    { \partial \mathbf{v}^T \mathbf{o} \over \partial o_1}   
  =   {\hat{y} - y \over \hat{y}(1-\hat{y}) }   \hat{y}(1-\hat{y})   v_1   
  =  (\hat{y}-y) v_1 
  $$
  <!-- \end{align} -->

<!-- - Thus, the gradient of $E$ over $\hat{y}$, i.e., $\hat{y}-y$ is backpropagated to $o_1$ by a factor of $v_1$.  -->

- Expand the second term
  $${ \partial \mathbf{o} \over \partial w_{1,1}} 
  = {\partial \phi(\mathbf{w}_1 \mathbf{x}) \over \partial \mathbf{w}_1 \mathbf{x}} 
    { \partial \mathbf{w}_1^T \mathbf{x} \over \partial w_{1,1}}
  = o_1(1-o_1) x_1
  $$

- Put together: 
  ${ \partial E \over \partial w_{1,1}} 
        ={ \partial E \over \partial o_1}
        {\partial \mathbf{o} \over \partial w_{1,1}}
    = (\hat{y}-y) (v_1) \left ( o_1(1-o_1) \right ) x_1$
::::
:::

# Backpropagation: an example II
::: {.columns}
:::: {.column width=0.3}
![](figs/one_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.73}
- Generalize for all weights in $\mathbf{w}_1 = [w_{1,1}, w_{2,1}, \dots, w_{d,1}, b_1]^T$:
${\partial E \over \partial \mathbf{w}_1} = (\hat{y}-y) (v_1) \left ( o_1(1-o_1) \right ) \mathbf{x}$
(where $\mathbf{x}=[x_1, x_2, \dots, x_d, x_{d+1}]$ and $x_{d+1}=1$)
- Similarly for weights between input neurons and $o_2$. 
  ${\partial E \over \partial \mathbf{w}_2} = (\hat{y}-y) (v_2) \left ( o_2(1-o_2) \right ) \mathbf{x}$
  (Note the change from $v_1$ to $v_2$ and that from $o_1$ to $o_2$)
::::
:::

- Stack these two together into matrix form, and write $o_i (1-o_i)$ back to its more general form $\phi'(o_i)$ for any activation function $\phi(\cdot)$: 
  $$\begin{pmatrix}
  \horzbar & \partial E \over \partial \mathbf{w_1} & \horzbar \\
  \horzbar & \partial E \over \partial \mathbf{w_2} & \horzbar 
  \end{pmatrix}
  = 
  \begin{pmatrix}
    v_1  \\ v_2
  \end{pmatrix}
  (\hat{y}-y)
  \circ  
  \begin{pmatrix}
    o_1 (1-o_1) \\
    o_2 (1-o_2) \\
  \end{pmatrix}
  \mathbf{x}^T
  =    
  \begin{pmatrix}
    v_1  \\ v_2
  \end{pmatrix}
  (\hat{y}-y)
  \circ  
  \phi' (\mathbf{o})
  \mathbf{x}^T$$

  where $\circ$ is 


# Backpropagation: an example III
::: {.columns}
:::: {.column width=0.3}
![](figs/one_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.73}
- Let's think one more step further: If there is another layer before $\mathbf{x}$, what is $\partial E \over \partial x_1$?
- The error onto $x_1$ is a composition of the error onto $o_1$ propagated thru $w_{1,1}$ and the error onto $o_2$ propagated thru $w_{1,2}$. 
- Thus 
   \begin{align*}
    & {\partial E \over \partial x_1} = 
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})  
    {\partial \mathbf{v}^T \mathbf{o} \over o_1}
    {\partial o_1 \over x_1}
    + 
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})
    {\partial \mathbf{v}^T \mathbf{o} \over o_2}
    {\partial o_2 \over x_2} \\
    = &
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})
    {\partial \mathbf{v}^T \mathbf{o} \over o_1}
    {\partial o_1 \over \mathbf{w_1}^T \mathbf{x}}
    {\partial \mathbf{w_1}^T \mathbf{x}\over x_1}
    + 
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})  
    {\partial \mathbf{v}^T \mathbf{o} \over o_2}
    {\partial o_2 \over \mathbf{w_2}^T \mathbf{x}}
    {\partial \mathbf{w_2}^T \mathbf{x}\over x_2} \\
    = & 
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})
    v_1
    \phi'(o_1)
    w_{1,1}
    + 
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})  
    v_2
    \phi'(o_2)
    w_{1,2} \\
    = & 
    \Big ( w_{1,1}~~w_{1,2} \Big )
    \begin{pmatrix}
    v_1 \phi'(o_1) \\
    v_2 \phi'(o_2) \\
    \end{pmatrix} 
    {\partial E \over \partial \hat{y}}   
    \phi'(\hat{y})
   \end{align*} 

::::
:::


# Backpropagation: a more complex example II

# Generalized backpropagation

- Let the error/loss **propagated** to a neuron $j$ at layer $l$ is $\delta^{(l)}_j$. 

- Then the gradient of that error on a neuron $i$ at layer $l-1$ is 
$${\partial \delta^(l)_j \over \partial x_i^(l-1)} = \delta^(l)_j w_{i,j} \phi'(w_{i,j} x^{(l-1)}_i) x^{(l-1)}_i$$


# What if $o_1$ is fed into multiple neurons? 





-   New challenge: How to compute the gradient for neurons not directly
    connected to final output? Just find its "fair share" to cost
    function.

-   As an example, let cost function be the difference between
    prediction $\phi$ and label $y$.
    $$\nabla (J(\mathbf{w}_i)) = \underbrace{ \frac{\partial (\phi - y) }{\partial \mathbf{w}_i} \pause = \frac{\partial \phi }{\partial \mathbf{w}_i}}_{y \text{ has nothing to do with } \mathbf{w}_i}  = \pause
      \underbrace{ \frac{\partial \phi}{\partial {o}_i} }_{\substack{\text{layers 2/hidden} \\ \text{ to 3/output}}} \pause \cdot 
      \underbrace{ \frac{\partial {o}_i}{\partial \mathbf{w}_i}}_{\substack{\text{layers 1/input} \\ \text{ to 2/hidden}}}
      \label{eq:two_stage_derivative}$$

-   -.5em Because of composition in each layer
    ($\phi = g(\mathbf{u}^T \mathbf{o})$ and each
    $\underset{{\scriptscriptstyle i\in[1..p]}}{o_i} = f(\mathbf{w}_i^T \mathbf{x})$
    ), by expanding
    Eq. ([\[eq:two_stage_derivative\]](#eq:two_stage_derivative){reference-type="ref"
    reference="eq:two_stage_derivative"}), we have:
    $$\hskip -2em \frac{\partial \phi}{\partial \mathbf{w}_i} = 
     \frac{\partial g(\mathbf{u}^T \mathbf{o})}{\partial (\mathbf{u}^T\mathbf{o})}
     \frac{\partial (\mathbf{u}^T\mathbf{o})}{\partial {o}_i } \pause
     \frac{\partial f(\mathbf{w}_i^T \mathbf{x})}{\partial \mathbf{w}_i^T \mathbf{x}}
     \frac{\partial  \mathbf{w}_i^T \mathbf{x}}{\partial \mathbf{w}_i} = \pause 
     \overbrace{
         \underbrace{g'(\mathbf{u}^T \mathbf{o}) \cdot u_i}_{
                     \substack{\text{error propagated} \\ \text{from 2nd layer} }}  \pause 
                     \cdot f'(\mathbf{w}_1^T\mathbf{x}) 
      }^{\text{all scalars}}  
      \cdot 
         \underbrace{\mathbf{x}}_{\substack{\text{perceptron}\\ \text{algorithm!}}}$$

Backpropagation

-   Eq. ([\[eq:two_stage_derivative\]](#eq:two_stage_derivative){reference-type="ref"
    reference="eq:two_stage_derivative"}) tells us that in order to
    compute the gradient in the current layer, we must have the product
    of gradients from all forward (output-bound) layers in hand.

-   Weights of connections are updated from the output to the input,
    against the direction of feedforward.

-   It resembles that the cost function is propagated from the output
    layer to the input layer, layer by layer.

Gradient vanishing problem

-   Will the gradient get larger or smaller as backpropagation moves on?

-   The derivative of the activation function (e.g., sigmoid, hyporbalic
    tangent) usually yields of a value in $[-1,1]$.

-   When you multiple a number with another number in $[-1,1]$, it
    becomes smaller.

-   Hence, the gradient becomes smaller and smaller (vanishes) as we
    backpropagate toward the input layer.

-   It takes really long to update weights near the input layer.

-   Solution: LTSM, residual networks, etc.

Deep Learning
=============

Deep Learning

-   Feature extraction:

    -   Conventional ML: manually craft a set of features.

    -   Problem: Sometimes features are too difficult to be manually
        designed, e.g., from I/O system log.

    -   A (no-brainer) solution: let computers find it for us, even by
        brutal force.

-   Not all weights matter:

    -   There are more tasks that need function fitting beyond
        conventional classification and regression.

    -   Ex. producing a sequence (e.g., a sentence)

    -   Sometimes we use the network to get something else useful, such
        as word embedding.

    -   Maybe weights of only a small set of layers are what we need
        from training.

-   Equally important to network architecture, the training scheme also
    matters (not just simple pairs of feature/input vectors and labels).

CNN

-   Convolutional layer: imagine convolution as matching two
    shapes/sequences/strings

-   Pooling layer

-   ReLU layer

-   Fully connected layer (basically this is the regular ANN)

-   Avoid overfitting: dropout, stochastic pooling, etc.

-   implementation: text-cnn

-   Visualization of the output at layers:
    <http://cs231n.github.io/convolutional-networks/>

-   Do we use backpropagation to update weights in every layer?

-   Some layers are unsupervised!

Vanilla RNN

-   An RNN is just an IIR filter (are you also a EE major?):
    $$y[n] = \sum_{l=1}^N a_l y[n-l] + \sum_{k=0}^{M} b_k x[n-k]$$ where
    $x[i]$ (or $y[i]$) is the $i$-th element (a scalar) in the input (or
    output) sequence.

-   RNN allows the output of a neuron to be used as the input of itself
    (typically) or others. Typically,
    $\mathbf{s}_{t+1} = U\mathbf{x}_t + W\mathbf{s}_t$ where
    $\mathbf{s}_{t+1}$ and $\mathbf{s}_{t}$ are the output of the neuron
    at steps $s+1$ and $s$ respectively.

-   Unrolling/unfolding an RNN unit:

    ![image](figures/rnn_unrolling.png){width=".8\\textwidth"}

Neural language model in Elman network

-   Recall that a language model predicts the Probability of a sequence
    of (words, characters, etc. )

-   Because of the properties of conditional probability, we want the
    probability of next word, given a short history of the sequence:
    $P(w_{t+1} | w_{i}, i\in [t-k..t])$

-   Elman network/simple RNN. Three layers:

    -   Input layer is the concatenation $\mathbf{x}(t)$ of two parts:
        the current **sequence** (not just one element!!!)
        $\mathbf{w}(t) = [w_{t-k}, \dots, w_{t}]$, plus output from the
        hidden layer in previous step $\mathbf{s}(t-1)$.

    -   hidden/context layer:
        $\mathbf{s}(t) = f\left( \mathbb{X} \mathbf{x}(t) \right)$ where
        $\mathbb{X}$ is the matrix of weights from the input layer to
        hidden layer.

    -   Output layer: multiple neurons, one of which of the highest
        activation corresponds to the best prediction. Each neuron
        corresponds to one element in the sequence, e.g.,
        word/character/etc.

-   The new language model:
    $P(w_{t+1} | \mathbf{w}(t), \mathbf{s}(t-1)),$ predicting the next
    output word given a short history $\mathbf{w}(t)$ up to current step
    $t$ and the hidden layer up to previous step $t-1$.

Neural language models

-   Feedforward: "A neural Probabilistic Language Model", Beigio et al.,
    JMLR, 3:1137--1155, 2003

-   "Recurrent neural network based language model", Mikolov et al.,
    Interspeech 2010

-   Multiplicative RNN: "Generating Text with Recurrent Neural
    Networks", Sutskever, ICML 2011

LTSM and GRU

-   Motivations:

    -   An simply deep RNN can be unrolled into many many layers.
        Gradient vanishing is significant.

    -   We also want weights to be attenuated/gated based on the states.

-   LSTM and GRU

    -   Instead of layers, we have cells.

    -   LSTM: forget gate, input gate, and output gate. The 3 gates are
        computed from current input $\mathbf{x}(t)$ and output from
        previous cell $\mathbf{s}(t-1)$. Then we "make a choice" between
        using previous state $\mathbf{c}(t-1)$ and current input and use
        the choice and output gate to make the final output.

    -   GRU: simpler, just reset gate and update gate.

    -   <http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/>

    -   <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>

LSTM ![image](figures/LSTM_unrolled.png){width="\\textwidth"} Source:
Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to
Action Sequences, Mei et al., AAAI-16

Seq-to-seq learning

-   Let's go one more level up.

-   Instead of predicting the next element in an input sequence, can we
    produce the entire output sequence from the input sequence?

-   There could be no overlap between the two sequences, e.g., from a
    Chinese sentence to a German sentence.

-   Two RNNs: encoder and decoder

-   "Learning Phrase Representations using RNN Encoder--Decoder for
    Statistical Machine Translation", Cho et al., EMNLP 2014