---
header-includes:
  - \usepackage{algorithm2e}
  - \usepackage{algorithmic}
---

# Math notations

## Math notations
  ''Computer Science is no more a science about computers than astronomy is about telescopes.'' -- Edsger Dijkstra


## Set
  - In mathematics, a \emph{set} is a collection of distinct objects, which are called \emph{elements}. An element $A$ belonging to a set $B$ is denoted as $A\in B$, read as ‘’A in B.'' No two elements can be the same in a set. 
  - We use curly brackets to enclose all members of a set, e.g., $\{4, 2, 1, 3\}$. 
  - A set can be finite, infinite or even empty, i.e.,  $\emptyset$\footnote{Outside the US, people use $\varnothing$}. 
  - Special set notations: \begin{itemize}
    * $\mathbb{Z}$ for all integers, $\mathbb{Z^+}$ for all positive integers. 
    * $\mathbb{R}$ for all real numbers. And $\mathbb{Z^+}$ for? 
    * $[X..Y]$ all integers between $X$ and $Y$.
    * \emph{closed interval}: $[X, Y]$ means all real numbers $a$ such that $X\leq a \leq Y$. 
    * \emph{open interval}: $[X, Y]$ means all real numbers $a$ such that $X\leq a \leq Y$. 
  - We use the notation $\{x_i\}_{i=N}^M$ or $\{x_i\}_{i\in [N..M]}$ as a shorthand for the set $\{x_N, x_{N+1}, \dots, x_M\}$. 
  - \emph{Set-builder notation}: e.g., $A=\{2\cdot x | x \in \mathbb{Z}, x^2>7 \} = \{6, 8, 9, \dots \}$


## Set (cont.)
  - Operations on set:
    * Union: $A\cup B = \{x| x \in A\text{~or~}x\in B\}$, e.g., $\{1,2,3\}\cup \{4, 5,6 \} = \{1, 2,3,4,5,6\}$
    * Intersection: $A\cap B = \{x| x \in A\text{~and~}x\in B\}$, e.g., $\{1,2,3\}\cup \{5,6, 3 \} = \{3\}$
    * Difference\footnote{In other parts of the world: $A-B$}: $A \slash B = \{ x| x \in A\text{~and~}x\not \in B\}$, e.g., $\{1,2,3\}\slash \{5,6, 3 \} = \{1,2,5,6\}$
    * subset and superset: $A\subseteq B$ iff $\forall x \in A$, $x\in B$ holds. We say that $A$ is a \emph{subset} of $B$ and $B$ a \emph{superset} of $A$. 
   
    iff is read as ''if and only if'' while $\forall$ is read as ''for all.''

    * True subset or superset: $A\subset B$ iff $A\subseteq B$ and $B\slash A=\emptyset$. We say that $A$ is a \emph{true subset} of $B$ and $B$ a \emph{true superset} of $A$. 
    * Homework: Prove that $A\subseteq B$ iff $A\cup B = B$. 
  - Venn Diagram
  - Cartesian Product: $A\times B = \{(a,b)| a\in A \text{~and~} b\in B\}$.


## Sequence, Tuple and Vector
  - A \emph{sequence} is an ordered collection of objects in which repetitions are allowed. Note that in set, there is no order nor repetition. A sequence is also called an \emph{ordered list}. 
  - An n-\emph{tuple} is a sequence of $n$ elements, where $n$ is a non-negative integer. In other words, a tuple is a finite sequence. It is also used interchangeably with the term \emph{vector} in the context of this class.
  - We usually use sharp bracket or square bracket. E.g., $< 1,23,3 >$ $[x_i]_{i=N}^M$, or $[x_i]_{i\in [N..M]}$. 
  - For the sake of space, we usually use bold font to denote a vector and usually the vector name is related to name for each of its elements, e.g., 
  $\mathbf{X} = [x_1, \dots, x_N]$. 
  - The number of element in the vector is called its \emph{dimension}, denoted as $\dim(\mathbf{X})$ or $|\mathbf{X}|$.
  - It is also common to use a rightarrow over to denote a vector, e.g., $\overrightarrow X$.


## Functions
  - A function is a mapping from a non-empty set (called \emph{domain}, could be a Cartesian product) of numbers to a non-emptyset (called \emph{range}) of numbers. Remember, everything is a number in the computer. 
  - The map is one-to-one or many-to-one. But cannot be one-to-many. 
  - A function is denoted in the following form usually: $f: A\times B \mapsto C$ where $f$ is the function name, $A$ and $B$ are the arguments or parameters, and $C$ the output or return. The symbol $\mapsto$ is read as ''maps to.''
  - If the input is one variable, this function is called \emph{univariate}. If it has more than one, \emph{multivariate}, including \emph{bivariate}. 
  - For example, the sine function is $\sin: \mathbb{R} \mapsto [0,1]$. 
  - Function composition: $(g\circ f)(x) = g(f(x))$.

  
## Functions (cont.)
  - Inverse function: $f^{-1}: Y \mapsto X$ if $f: X\mapsto Y$ and, both $f$ and $f^{-1}$ are one-to-one mappings.
  - The ratio of change rate between the output of a multivariate function $f$ and one input $x$ is called the \emph{derivative}. In discrete domains, it is denoted as 
  $$
  {\frac{\partial f}{\partial x}} \bigg|_{f=f_n, x=x_n}  \equiv \frac{f_{n} - f_{n-1}}{x_{n}-x_{n-1}}
  $$
  where $n$ is the index for both inputs and outputs. 
  The big vertical bar is read as ''evaluated at''. Note that there is no \emph{analytical} expression of derivative in discrete domains. 
  - For the sake of space, people could apply a function on a vector, e.g., 
  $f(\mathbf{X}, \mathbf{Y}) = \mathbf{Z}$ where $\mathbf{X} = [x_1, \dots, x_N]$, $\mathbf{Y} = [y_1, \dots, y_N]$, $\mathbf{Z} = [z_1, \dots, z_N]$, and $\forall i\in [1..N]$, we have $f(x_i, y_i) = z_i$.  


## Linear Algebra
  - Dimension: The dimension of a matrix is $N\times M$ if it has $N$ rows (horizontal) and $M$ columns (vertical). 
  - Matrix multiplication (not to be confused with \emph{element-wise multiplication}).  Why is matrix multiplication defined in this way? 
  - Dot product of vectors:  $\sum_{i=1}^N x_i\cdot y_i = \mathbf{X}\cdot \mathbf{Y}$, short as $\mathbf{X} \mathbf{Y}$. 
  - Euclidean norm ($L^2$ norm): $\left\| \boldsymbol{X} \right\| := \sqrt{x_1^2 + \cdots + x_n^2}$ where $\mathbf{X} = [x_1, \dots, x_n]$.
  - Transpose: $\mathbf{X} = \begin{bmatrix} x_1 \\ x2 \end{bmatrix}$, $\mathbf{X}^T = [x_1, x_2]$. 
  - Trace of a matrix: $\mathbf{Tr}(A) = [a_{1,1}, \dots, a_{N,N}]$ for a square matrix $A$ of dimension $N\times N$. 


## Linear Algebra II
  - Linear systems as matrixes. For example
    $$
        \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}  = \begin{bmatrix} a_{1,1} & a_{1,2} \\ a_{2,1} & a_{2,2} \end{bmatrix} \times \begin{bmatrix} y_1 \\ y_2 \end{bmatrix}
    $$
 
    is equivalent to 
 
    $$
        \begin{array}{c} 
        x_1 = a_{1,2} y_1 + a_{1,2} y_2\\
        x_2 = a_{2,1} y_1 + a_{2,2} y_2\\
        \end{array}
    $$
  
    which can also be written as 
    $\mathbf{X} = \mathbf{A} \mathbf{Y}$ where $A = \begin{bmatrix} a_{1,1} & a_{1,2} \\ a_{2,1} & a_{2,2} \end{bmatrix}$.
  
  - Identity matrix (or einheit matrix): $I_n = \begin{bmatrix}
    1 & 0 & 0 & \cdots & 0 \\
    0 & 1 & 0 & \cdots & 0 \\
    0 & 0 & 1 & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \cdots & 1 \end{bmatrix}$


## Linear Algebra III
  - Inverse of a matrix: $\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}_n$. A non-inversible matrix is called a singular matrix. 
  - Determinant of a matrix: A matrix $A$ is inversible iff $\mathbf{det}(A) \not = 0$ (and many other equivalence). 
  - Eigenvalue and Eigenvector: A eigenvalue $\lambda$ for a square matrix $\mathbf{A}$ is a scalar such that $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ where $\lambda$ is another vector, called the \emph{eigenvector}. A matrix can have many eigenvalues, each of which is paired with one eigenvector. A inversible matrix has them. 


# Introduction to Machine Learning


## Mathematical formulation of machine learning
  - The task of statistical ML is to build a numerical predictive \emph{model/estimator}, which is a function $f:\boldsymbol{X} \mapsto \boldsymbol{y}$ . 
  - Three kinds of machine learning: 
  - Function approximation/fitting: 
    1. Supervised learning: fit the function $f$ given pairs of $\boldsymbol{X}$, called a training input (or feature vector if not raw) and $\boldsymbol{y}$, called the target/label.
    2. Reinforcement learning: fit the function $f$ given pairs of $\boldsymbol{X}$, and $\boldsymbol{y}$, which is now called a value/cost function, defined by the interaction between the agent and the environment. 
    3. Unsupervised learning: learn to find the function $f$ given only $\boldsymbol{X}$. No ground truth. Not function fitting. 
  - Deep learning: When the function $f$ is highly complicated, that people usually use a deep neural network to represent. So there can be Deep X learning. 


## Supervised Learning
  - In \emph{supervised learning}, a pair (2-tuple) of an input/feature vector and a target form a (training) \emph{ sample}. A finite set of samples $\{(\mathbf{X_1}, y_1), \dots, (\mathbf{X_N}, y_N)\}$ form a \emph{training set}, where each $\mathbf{X_i}$ $\in \mathbb{R}^n$ ($i \in [1..N]$) is a feature vector while each $y_i$ is a target. 
  - If the set $\boldsymbol{y}$ is discrete, e.g., $\boldsymbol{y}=\{+1, -1\}$,  we call $f$ a \emph{classifier}. Otherwise, a \emph{regressor}, e.g., $f: \mathbb{R}^n \mapsto \mathbb{R}$.
  - Without losing generality, in this class a target is a real \emph{scalar} while a feature vector is an $n$-dimensional real vector, i.e., $\boldsymbol{X} \subseteq \mathbb{R}^n$ and $\boldsymbol{y}\subseteq \mathbb{R}$. 
  - Given a training set, an ML algorithm will find such a function $f$, usually through solving a numerical optimization problem, to minimize the cost function, e.g., RMSE. 
  - Given a new label-less sample $\mathbf{X}_{new}$, the prediction is $f(\mathbf{X}_{new})$.
  - The $f$ is also called a \emph{hypothesis}. And we can have many such hypotheses, forming the \emph{hypothesis space}.


## Features
  ![](figs/p1.pdf){width=60%}

  - A sample has a feature vector to numerically represent it. 
  - For example, if you want to build a classifier to distinguish apples and banana, what features do you plan to use?
  - Roundness, color, etc. 
  - Features can be non-human-readable, e.g., Google's Word2Vec. 
  - Deep learning is an approach to find features, or in the buzz word,  the \emph{abstract} of data. 


# Linear Classifiers


## The hyperplane
  ![](figs/p2.pdf){width=60%}

  - Now, let's begin our journey on supervised learning. 
  - Suppose we have a line going thru points $(0, w_1)$ and $(w_2, 0)$ (which are the \emph{intercepts}) in a 2-D vector space spanned by two orthogonal bases $x_1$ and $x_2$.
  - The equation of this line is $x_1 w_1 + x_2 w_2 - w_1 w_2 =0$. 


## The hyperplane (cond.)
  - Let 
    $$
      \mathbf{x} = \begin{pmatrix}
                      x_1 \\ x_2 \\ 1 
                   \end{pmatrix}
    $$  
    and
    $$
      \mathbf{w} = \begin{pmatrix}
                      w_1 \\ w_2 \\ -w_1w_2
                   \end{pmatrix}
    $$. (By default, all vectors are column vectors.)\\ 
    Then the equation is rewritten into vector form: $\mathbf{x}^T \cdot \mathbf{w} = 0$.\\ 
    For space sake, $\mathbf{x}^T \mathbf{w} = \mathbf{x}^T \cdot \mathbf{w}$. 
  - Expand to $n$-dimension. 
    $$
      \mathbf{X} = \begin{pmatrix}
                      x_1 \\ x_2 \\ \vdots \\ x_n \\ 1 
                   \end{pmatrix}
    $$  
    and
    $$\mathbf{W} = \begin{pmatrix}
                      w_1 \\ w_2 \\ \vdots \\w_n \\ -w_1w_2
                   \end{pmatrix}
    $$. 

    Then $\mathbf{X}^T \cdot \mathbf{W} = 0$, denoted as the \emph{hyperplane} in $\mathbb{R}^n$.
  - In our class, the $[x_1, x_2, \dots, x_n]$ is a feature vector. 
  - The last term of $\mathbf{W}$ is often called a \emph{bias} or a \emph{threshold}.


## Binary Linear Classifier
  - A binary linear classier is a function $f(X)=\mathbf{W} \mathbf{X}$, such that
  $$
    \begin{cases}
      \mathbf{W}^T\mathbf{X} >0 & \forall X\in C_1\\
      \mathbf{W}^T\mathbf{X} <0 & \forall X\in C_2
    \end{cases}
  $$
  where $C_1$ and $C_2$ are the two classes. Note that the $\mathbf{X}$ has been augmented with 1 as mentioned before. 
  - Finding such an $\mathbf{W}$ is the \emph{learning}. 
  - Using the function $f$ to make decision is called \emph{test}. Given a new sample whose augmented feature vector is $\mathbf{X}$, if $\mathbf{W}^T\mathbf{X} >0$, then we classify the sample to class $C_1$. Otherwise, class $C_2$. 
  - Example. Let $\mathbf{W}^T = (2, 4, -8)$, what's the class for new sample $\mathbf{X}= (1,1,1)$ ($1$ is augmented)? 
  - $\mathbf{W}^T\mathbf{X} = -2 <0$. Hence the sample of feature value $(1,1)$ belongs to class $C_1$.


## Solving inequalities: the simplest way to find the $\mathbf{W}$
  - Let's look at a case where the feature vector is 1-D. 
  - Let the training set be $\{(4, C_1), (5, C_1), (1, C_2), (2, C_2)\}$. Their augmented feature vectors are: $X_1=(4, 1)^T$, $X_2=(5, 1)^T$, $X_3=(1, 1)^T$, $X_4=(2, 1)^T$.
  - Let $\mathbf{W}^T = (w_1, w_2)$. In the training process, we can establish 4 inequalities: 
  $$
    \begin{cases}
      4 w_1+ w_2 & >0  \\
      5 w_1+ w_2 & >0  \\
        w_1+ w_2 & <0  \\
      2 w_1+ w_2 & <0  \\
    \end{cases}
  $$  
  - We can find many $w_1$ and $w_2$ to satisfy the inequalities. But, how to pick the best? 
  - But let's talk about one more algorithm before defining the cost function.


## Normalized feature vector
  - I am lazy. I hate to write two cases. 
  - A correctly classified sample $(\mathbf{X_i}, y_i)$ shall satisfy the inequality $\mathbf{W}_i^T\mathbf{X} y_i > 0$.  (The $y_i$ flips the direction of the inequality. )
  - \textit{normalize} the feature vector: $\mathbf{X}_i y_i$ for $y_i\in  \{+1, -1\}$.
  - Example: Four samples, where\\
    $\mathbf{x}'_1= (0, 0 )^T$, $\mathbf{x}'_2= (0, 1)^T$, $\mathbf{x}'_3= (1, 0)^T$, $\mathbf{x}'_4= (1, 1)^T$
    $y_1=1, y_2= 1, y_3= -1, y_4= -1$
    First, let's augment and normalize them: 
    $\mathbf{x}_1= (0, 0, 1)^T$, $\mathbf{x}_2= (0, 1,1 )^T$, $\mathbf{x}_3= (-1, 0, -1)^T$, $\mathbf{x}_4= (-1, -1, -1)^T$
  - Please note that the term ''normalized'' could have different meanings in different context of ML.  


# least-squared and Fisher's criteria


## Gradient
  - The partial derivative of a multivariate function is a vector called the gradient, representing the derivatives of a function on different directions. 
  - For example, let $f(\mathbf{x}) = x_1^2 + 4x_1 + 2x_1x_2 + 2x_2^2 + 2x_2 + 14$. $f$ maps a vector $\mathbf{x} = (x_1, x_2)^T$ to a scalar. 
  - Then we have 
    $$\nabla f  = \frac{\partial f}{\partial \mathbf{x}}  = 
      \begin{pmatrix}
        \frac{\partial f}{\partial x_1}\\
        \frac{\partial f}{\partial x_2}
      \end{pmatrix}
    =   
      \begin{pmatrix}
        2x_1+ 2x_2 -4 \\
        4x_2 + 2x_1 + 2
      \end{pmatrix}$$
  - The gradient is a special case of \emph{Jacobian matrix}. (see also: \emph{Hessian matrix} for second-order partial derivatives.)
  - A \emph{critical point} or a \emph{stationary point} is reached where the derivative is zero on any direction.
    * local extremum 
      - local maximum
      - local minimum
    * saddle point
  - if a function is convex, a local minimum/maxinum is the \emph{global minimum/maximum}. 


## Find the linear classifier using an optimization way I
  - Two steps here:
    * Define a cost function to be minimized (The learning is the about minimizing the cost function)
    * Choose an algorithm to minimize (e.g., gradient, least squared error etc. )
  - One intuitive criterion can be the sum of error square: 
  $$ J(\mathbf{W}) = \sum_{i=1}^N (\mathbf{W}^T\mathbf{x}_i -y_i)^2 = \sum_{i=1}^N (\mathbf{x}_i^T \mathbf{W} -y_i)^2 $$


## Find the linear classifier using an optimization way II
  - Minimizing $J(\mathbf{W})$ means (Convexity next time.) $\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}} = 2\sum\limits_{i=1}^N \mathbf{x}_i (\mathbf{x}_i^T \mathbf{W} - y_i) = (0, \dots, 0)^T$ 
  - Hence, 
  $\sum\limits_{i=1}^N \mathbf{x}_i \mathbf{x}_i^T \mathbf{W} = \sum\limits_{i=1}^N \mathbf{x}_i y_i$
  - The sum of a column vector multiplied with a row vector produces a matrix.
  $$
    \sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^T = 
      \begin{pmatrix}
        \rule[-1ex]{0.5pt}{2.5ex}& \rule[-1ex]{0.5pt}{2.5ex}& & \rule[-1ex]{0.5pt}{2.5ex}\\
        \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_N \\
        \rule[-1ex]{0.5pt}{2.5ex}& \rule[-1ex]{0.5pt}{2.5ex}& & \rule[-1ex]{0.5pt}{2.5ex}
      \end{pmatrix}
      \begin{pmatrix}
        \rule[.5ex]{2.5ex}{0.5pt}& \mathbf{x}_1^T & \rule[.5ex]{2.5ex}{0.5pt}\\
        \rule[.5ex]{2.5ex}{0.5pt}& \mathbf{x}_2^T & \rule[.5ex]{2.5ex}{0.5pt}\\
            &       \vdots        &   \\
        \rule[.5ex]{2.5ex}{0.5pt}& \mathbf{x}_N^T & \rule[.5ex]{2.5ex}{0.5pt}\\
      \end{pmatrix}
      =\mathbb{X}^T \mathbb{X}
  $$


## Find the linear classifier using an optimization way II
  - $$\sum_{i=1}^N \mathbf{x}_i y_i = 
      \begin{pmatrix}
        \rule[-1ex]{0.5pt}{2.5ex}& \rule[-1ex]{0.5pt}{2.5ex}& & \rule[-1ex]{0.5pt}{2.5ex}\\
        \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_N \\
        \rule[-1ex]{0.5pt}{2.5ex}& \rule[-1ex]{0.5pt}{2.5ex}& & \rule[-1ex]{0.5pt}{2.5ex}
      \end{pmatrix}
      \begin{pmatrix}
        y_1 \\
        y_2 \\
        \vdots   \\
        y_N \\
      \end{pmatrix}
      =\mathbb{X}^T \mathbf{y}
$$
- $\mathbb{X}^T\mathbb{X}\mathbf{W} = \mathbb{X}^T \mathbf{y}$
- $(\mathbb{X}^T\mathbb{X})^{-1}\mathbb{X}^T\mathbb{X}\mathbf{W} = (\mathbb{X}^T\mathbb{X})^{-1}\mathbb{X}^T \mathbf{y}$
- $\mathbf{W} 
 = (\mathbb{X}^T\mathbb{X})^{-1}\mathbb{X}^T \mathbf{y}$


## Gradient descent approach
  Since we define the target function as $J(\mathbf{W})$, finding $J(\mathbf{W})=0$ or minimizing $J(\mathbf{W})$ is intuitively the same as reducing $J(\mathbf{W})$ along the gradient. The algorithm below is a general approach to minimize any multivariate function: changing the input variable  proportionally to the gradient.
  \begin{columns}
    \begin{column}{.6\textwidth}
      \begin{algorithm}[H]
        \caption{pseudocode for gradient descent approach}
        \label{alg:seq}
        \textbf{Input}: an initial $\mathbf{w}$, stop criterion $\theta$, a learning rate function $\rho(\cdot)$, iteration step $k=0$
        \begin{algorithmic}[1]
          \WHILE{$\nabla J(\mathbf{w}) > \theta$}
            \STATE $\mathbf{w}_{k+1} := \mathbf{w}_k - \rho(k) \nabla J(\mathbf{w}) $
            \STATE $k := k+1 $
          \ENDWHILE
        \end{algorithmic}
      \end{algorithm} 
    \end{column}
    \begin{column}{.36\textwidth}
      \includegraphics[width=\textwidth]{figs/Gradient_descent.png}
    \end{column}
  \end{columns}

## Gradient descent approach (cond.)
  In many cases, the $\rho(k)$'s amplitude (why amplitude but not the value?)  decreases as $k$ increases, e.g., $\rho(k) = \frac{1}{k}$, in order to shrink the adjustment.Also in some cases, the stop condition is $\rho(k)\nabla J(\mathbf{w}) > \theta$. The limit on $k$ can also be included in stop condition -- do not run forever. 


## Fisher's linear discriminant
  - What really is $\mathbf{w}^T x$? $\mathbf{w}$ is perpendicular to
    the hyper panel [^3]
  - $\mathbf{w}^T \mathbf{x}$ is the *projection* of the point
    $\mathbf{x}$ on the decision panel.
  - Intuition in a simple example: for any two points
    $\mathbf{x}_1 \in C_1$ and $\mathbf{x}_2\in C_2$, we want
    $\mathbf{w}^T \mathbf{x}_1$ to be as different from
    $\mathbf{w}^T \mathbf{x}_1$ as possible, i.e.,
    $\max (\mathbf{w}^T \mathbf{x}_1 - \mathbf{w}^T \mathbf{x}_2)^2$.
    \[Fig. 4.6, Bishop book\]
  - For binary classification, intuitively, we want the projections of
    the same class to be close to each other (i.e., the smaller
    $\tilde{s}_1$ and $\tilde{s}_2$ the better) while the projects of
    different classes to be apart from each other (i.e., the larger
    $(\tilde{m}_1 - \tilde{m}_2)^2$ is better).
  - That means $$\max J(\mathbf{w}) = 
      \frac{(\tilde{m}_1 - \tilde{m}_2)^2}
      {\tilde{s}_1^2 + \tilde{s}_2^2}$$ where
    $\tilde{m}_i = \frac{1}{|C_i|} \sum\limits_{\mathbf{x} \in C_i} \mathbf{w}^T\mathbf{x}$
    and
    $\tilde{\mathbf{s}}^2_i = \sum\limits_{\mathbf{x}\in C_i} (\mathbf{w}^T \mathbf{x} - \tilde{m}_i)^2$
    are the mean and the variance of the projection of all samples
    belonging to Class $i$ on the decision panel, respectively.


## Fisher's (cond.)
  - between-class scatter:
    $(\tilde{m}_1 - \tilde{m}_2)^2 = (\mathbf{w}^T (\mathbf{m_1} - \mathbf{m_2}))^2 = 
      \mathbf{w}^T  (\mathbf{m_1} - \mathbf{m_2}) (\mathbf{m_1} - \mathbf{m_2}) ^T \mathbf{w}$
    where
    $\mathbf{m}_i = \frac{1}{|C_i|} \sum\limits_{\mathbf{x} \in C_i} \mathbf{x}$
  - within-class scatter:
    $\tilde{\mathbf{s}}^2_i = \sum\limits_{\mathbf{x}\in C_i} (\mathbf{w}^T \mathbf{x} - \tilde{m}_i)^2 
      =\sum\limits_{\mathbf{x}\in C_i} (\mathbf{w}^T \mathbf{x} - \mathbf{w}^T\mathbf{m}_i)^2 = 
       \mathbf{w}^T [  \sum\limits_{\mathbf{x}\in C_i}(\mathbf{x - m}_i) (\mathbf{x - m}^T_i)]
       \mathbf{w}$
  - Denote
    $\mathbf{S_w} = \tilde{\mathbf{s}}^2_1 + \tilde{\mathbf{s}}^2_2$ and
    $\mathbf{S}_B = (\mathbf{m_1} - \mathbf{m_2}) (\mathbf{m_1} - \mathbf{m_2}) ^T$.
    We have
    $$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$.
    This expression is known as *Rayleigh quotient*.
  - To maximize $J(\mathbf{w})$, the $\mathbf{w}$ must satisfy
    $\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_w \mathbf{w}$.
  - Hence
    $\mathbf{w} = \mathbf{S}_w^{-1} (\mathbf{m}_1 - \mathbf{m}_2)$.
    (Derivate saved.)