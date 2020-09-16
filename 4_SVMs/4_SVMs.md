---
title: | 
         CS 474/574 Machine Learning \
         4. Support Vector Machines (SVMs)
author: |
          Prof. Dr. Forrest Sheng Bao \
          Dept. of Computer Science \
          Iowa State University \
          Ames, IA, USA \
date:   \today
header-includes: |
     \usepackage{amssymb}
     \usefonttheme[onlymath]{serif}
     \usepackage[vlined,algoruled,titlenotnumbered,linesnumbered]{algorithm2e}
     \usepackage{algorithmic}
     \setbeamercolor{math text}{fg=green!50!black}
     \setbeamercolor{normal text in math text}{parent=math text}
     \newcommand*{\vertbar}{\rule[-1ex]{0.5pt}{2.5ex}}
     \newcommand*{\horzbar}{\rule[.5ex]{2.5ex}{0.5pt}}

classoption:
- aspectratio=169
---

# 

SVM is a kind of linear classifiers. But a unique type of. 

# All samples are equal. But some samplers are equaler. 

- Let's first see a demo of a linear classifier for linearly separable cases. Pay attention to the prediction outcome. 
- Think about the error-based loss function for a classifier: $\sum_i (\hat{y} - y )^2$ where $y$ is the ground truth label and $\hat(y)$ is the prediction. 
- If $y=+1$ and $\hat{y} = +1.5$, should the error be 0.25 or 0 (because properly classified)? 

# The perceptron algorithm

-   Recall earlier that a sample $(\mathbf{x}_i, y_i)$ is correctly
    classified if $\mathbf{w}^T \mathbf{x}_i y_i > 0$ .

-   Let's define a new cost function to be minimized:
    $J(\mathbf{w}) = \sum\limits_{x_i \in \mathcal{M}} - \mathbf{w}^T \mathbf{x}_iy_i$
    where $\mathcal{M}$ is the set of all samples misclassified
    ($\mathbf{W}^T \mathbf{X}_i y_i < 0$).

-   Then,
    $\nabla J(\mathbf{w}) =  \sum\limits_{\mathbf{x}_i \in \mathcal{M}} - \mathbf{X}_iy_i$
    (because $\mathbf{w}$ is the coefficients.)

-   Only those misclassified matter! 

-   Batch perceptron algorithm: In each batch, computer
    $\nabla J(\mathbf{w})$ for all samples misclassified using the same
    current $\mathbf{w}$ and then update.

# Single-sample perceptron algorithm 

-   Another common type of perceptron algorithm is called single-sample
    perceptron algorithm.

-   Update $\mathbf{w}$ whenever a sample is misclassified.

    1.  Initially, $\mathbf{w}$ has arbitrary values. $k=1$.

    2.  In the $k$-th iteration, use sample $\mathbf{x}_j$ such that
        $j = k \mod n$ to update the $\mathbf{w}$ by:
        $$        \mathbf{W}_{k+1} = \begin{cases}
                                 \mathbf{W}_k + \rho \mathbf{X}_j y_j & \text{, if } \mathbf{W}_j^T \mathbf{X_j} y_j \leq 0, \text{~(wrong prediction)} \\
                                 \mathbf{W}_k  & \text{, if } \mathbf{W}_j^T \mathbf{X_j} y_j > 0 \text{~(correct classification)}
                               \end{cases}        $$
        where $\rho$ is a constant called **learning rate**.

    3.  The algorithm terminates when all samples are classified
        correctly.

-   Note that $\mathbf{x}_k$ is not necessarily the $k$-th training
    sample due to the loop.

# An example of single-sample preceptron algorithm

:::::::::::::: columns
::: {.column width="50%"}
- Feature vectors and labels: 

  - $\mathbf{x}'_1= (0, 0)^T$, $y_1=1$
  - $\mathbf{x}'_2= (0, 1)^T$, $y_2=1$
  - $\mathbf{x}'_3= (1, 0)^T$, $y_3=-1$
  - $\mathbf{x}'_4= (1, 1)^T$, $y_4=-1$

 - First, let's augment them and multiply with the labels: 
   - $\mathbf{x}_1y_1 = (0, 0, 1)^T$, 
   - $\mathbf{x}_2y_2= (0, 1,1 )^T$, 
   - $\mathbf{x}_3y_3= (-1, 0, -1)^T$
   - $\mathbf{x}_4y_4= (-1, -1, -1)^T$
:::

::: {.column width="50%"}

0. Begin our iteration. Let $\mathbf{w}_1= (0,0,0)^T$ and $\rho=1$.

1.  $\mathbf{W}_1^T \cdot \mathbf{x}_1 y_1= 
    \begin{pmatrix}
    0 & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
    0 \\
    0 \\
    1
    \end{pmatrix} = 0 \leq 0$. Need to update $\mathbf{W}$:
    $\mathbf{W}_2 = \mathbf{W}_1 + \rho \cdot \mathbf{x}_1 y_1 = 
    \begin{pmatrix}
    0 \\
    0 \\
    0
    \end{pmatrix}
    +
    \begin{pmatrix}
    0 \\
    0 \\
    1
    \end{pmatrix} = 
    \begin{pmatrix}
    0 \\
    0 \\
    1
    \end{pmatrix}$
:::
::::::::::::::

2.  $\mathbf{W}_2^T \cdot \mathbf{x}_2 y_2= 
    \begin{pmatrix}
    0 & 0 & 1
    \end{pmatrix}
    \begin{pmatrix}
    0 \\
    1 \\
    1
    \end{pmatrix} = 1 >0$. No updated need. But since $\mathbf{w}$ so far does not
    classify all samples correctly, we need to keep going. Just let
    $\mathbf{w}_3 = \mathbf{w}_2$.

# An example of preceptron algorithm (cond.)

:::::::::::::: columns
::: {.column width="50%"}

Example in [perceptron.ipynb](./perceptron.ipynb)

14.   In the end, we have $\mathbf{W}_{14} = 
     \begin{pmatrix}
     -3 \\
     0 \\
     2
    \end{pmatrix}$, let's verify how well it works

-   $$\begin{cases}
             \mathbf{w}_{14}\cdot \mathbf{x}_1 y_1 &= 1 > 0 \\
             \mathbf{w}_{14}\cdot \mathbf{x}_2 y_2&= 1 > 0 \\
             \mathbf{w}_{14}\cdot \mathbf{x}_3 y_3&= 1 > 0 \\
             \mathbf{w}_{14}\cdot \mathbf{x}_4 y_4&= 1 > 0
             \end{cases}$$
:::

::: {.column width="50%"}


-   Mission accomplished!

-   Note that the perceptron algorithm will not *converge* unless the
    data is linearly separable.

-   What is $\mathbf{w}$ exactly? A linear composition of all training samples!

-   Do all samples contribute to $\mathbf{w}$? Not really! 

:::
::::::::::::::

