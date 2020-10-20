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
    \usepackage{amssymb}
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
classoption:
- aspectratio=169
---

# Teaching 
- Github, slides 
- Links are now added 
- spoon-fed vs. hint-driven 
- To google or not to google
- fish vs. fishing
- hw, exploratory problems 

# Why artificial neural networks (ANNs) work

-   Supervised or Reinforcement learning is about function fitting.

-   To get the function, the analytical form sometimes doesn't matter.

-   As long as we can mimic/fit it accurately enough, it's good.

-   An ANN is a magical structure that can mimic any function \[Fig.
    5.3, Bishop book\], if the ANN is "complex" enough. 
    -- Known as "Universal Approximation." 

# One neuron/preceptron
::: {.columns}
:::: {.column width=0.4}
![](figs/one_neuron_2.pdf)
::::
:::: {.column width=0.6}
-   The inputs and output of a **neuron** (also called a **preceptron** or a **node**) follow this mapping: 
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

# A preceptron vs. the preceptron algorithm

-   Why is one algorithm seen earlier called  the "preceptron"
    algorithm?

-   Because $\phi(\mathbf{w}^T \mathbf{x})$ is exactly one
    neuron/preceptron in an ANN.

-   Frank Rosenblatt published his preceptron algorithm in 1962 titled
    "Principles of Neurodynamics: Perceptrons and the Theory of Brain
    Mechanisms."

-   Linearly separable cases only! It cannot even do XOR.

-   Therefore, Marvin Minsky jumped to the conclusion that ANNs were
    useless. \[Perceptrons, Marvin Minsky and Seymour Papert, MIT Press,
    1969\]

-   However, Minsky is an AAAI fellow but not a prophet.

-   If we expand a preceptron into layers of preceptrons, we get an **artificial neural network (ANN)** or an **multi-layer preceptron (MLP)**, which is much more powerful, and for sure, can mimic XOR. 

# Let's try to insert two neurons between inputs and output. 

::: {.columns}
:::: {.column width=0.35}
![](figs/one_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.72}

<!-- - Layer-2 neurons produce outputs independently and according to Eq. \ref{eq:one_neuron}. -->

- Relationship between input neurons and the two "mid-layer" neurons:
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

- Rewrite into matrix form: 
$\phi(\mathbb{W}^T\mathbf{x}) = \mathbf{o}$
where 
$\mathbb{W} = \begin{pmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2} \\
\vdots & \vdots \\
b_1 & b_2 
\end{pmatrix}
= \begin{pmatrix}
\rotatebox[origin=c]{270}{1st mid neuron} 
\rotatebox[origin=c]{270}{weights for} & 
& 
\rotatebox[origin=c]{270}{2nd mid neuron} 
\rotatebox[origin=c]{270}{weights for} &
& 
\cdots 
  \end{pmatrix}$
::::
:::

- Similarly, between "mid-layer" and final output:
  $\hat{y}=\phi(\mathbb{V}^T\mathbf{o})$. In this example, $\mathbb{V}$ has only one column. Why? 

- How to expand $\mathbb{W}$ if there are more than two neurons in the middle?

# Adding more neurons into the middle layer

::: {.columns}
:::: {.column width=0.4}
![](figs/larger_hidden_layer.pdf){width=100%}
::::
:::: {.column width=0.62}

- Because each neuron in middle layer is a linear classifier/regressor, adding a neuron in middle layer means inseting a new column of weights into $\mathbb{W}$. 

- For the example neural network to the left: 
    $\phi(\mathbb{W}^T\mathbf{x}) = \mathbf{o}$
    where 
    $\mathbb{W} = \begin{pmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    w_{2,1} & w_{2,2} & w_{2,3}\\
    \vdots & \vdots & \vdots  \\
    b_1 & b_2 & b_3
    \end{pmatrix}$

- And again, from middle layer to output: 
  $\hat{y}=\phi(\mathbb{V}^T\mathbf{o})$, where 
  $\mathbb{V} = 
  \begin{pmatrix}
  v_1\\v_2\\v_3
  \end{pmatrix}$

::::
:::

# Layers and feed-forward

- An ANN consists multiple **layers** of preceptrons.

- Types of layers: 
    * neurons have no input form the **input layer**
    * neurons do not output into other neurons form the **output layer**
    * **hidden layers**: between the two special layers above.

- For hidden-layer neurons, if they share the same inputs (not necessarily the input of the ANN), then they belong to the same layer. Mathematically, all neurons, whose outputs $\mathbf{o}$ are resulted from the same transform $\mathbf{o} = \phi(\mathbb{W}^T\mathbf{x})$  where $\mathbf{x}$ is the outputs of previous-layer neurons, belong to the same layer. 

- Each layer can have any number of neurons. Minimal is one. 

- The transform $\mathbf{o} = \phi(\mathbb{W}^T\mathbf{x})$ is called **feedforward** where $\mathbf{x}$ is "fed" from previous layer to current layer to produce $\mathbf{o}$. It is a basic algorithm for an ANN to yield outputs. 

- Apply feedforward again and again, you can create a complex ANN: 
  $\phi\left ( \mathbb{W}^{(l)T} \cdots \phi \left( \mathbb{W}^{(2)T} \phi(\mathbb{W}^{(1)T}\mathbf{x}) \right)  \right )$ where $\mathbb{W}^{(i)}$ is the weights from layer ($i-1$) to layer $i$. 

# Example ANNs

::: {.columns}
:::: {.column width=0.65}
![](figs/two_hidden_layers.pdf){width=100%}
::::
:::: {.column width=0.35}
![](figs/XOR.png){width=100%}
::::
:::

# Quiz 
In order to approximate the relation $h={1\over 2}gt^2$

- How many neurons are needed in the input layer?

- How many neurons are needed in the output layer? 

- If the activation function is logistic function, are hidden layers necessary? Why? 

- How many hidden neurons are needed at least? 

# 

Something terminology

-   activation: the output of a neuron, which is applying an activation
    function onto the a weighted sum of its inputs, denoted as
    $f(\mathbf{w}^T\mathbf{x})$.

-   Input/Hidden/Output layer

-   Forward or forward propagation

-   connection/synapse

Gradient descent on multilayer perceptrons

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