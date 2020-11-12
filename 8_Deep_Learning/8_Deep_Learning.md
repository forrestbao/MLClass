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
    \usepackage{amssymb,mathtools,blkarray,bm}
        \usefonttheme[onlymath]{serif}
    \usepackage[vlined,algoruled,titlenotnumbered,linesnumbered]{algorithm2e}
    \usepackage{algorithmic}
    \setbeamercolor{math text}{fg=green!50!black}
    \setbeamercolor{normal text in math text}{parent=math text}
    \newcommand*{\vertbar}{\rule[-1ex]{0.5pt}{2ex}}
    \newcommand*{\horzbar}{\rule[.5ex]{2ex}{0.5pt}}
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
    \usepackage{graphicx}
classoption:
- aspectratio=169
---

# 

Images that were not created by the instructor but from the web 
are cited in the Markdown source code in the syntax: 
```
![a web image](URL)
```

# Deep learning (DL)

- Deep neural networks (DNNs): extensive amounts of layers. 

![a web image](https://user-images.githubusercontent.com/17570785/50308846-c2231880-049c-11e9-8763-3daa1024de78.png)


# Why is DL something relatively new? 

- DNNs use the same algorithms as any (shallow) ANNs: feedforward and backpropagation

- But DL needs two new sauces: "Big Data" and "Big Processor"

- Extensive amounts of layers means extensive amounts of transfer matrixes (weights) that need Big Data to train 

- It wasn't feasible until massive data was digitized (how many pictures were digital before iPhone?)

- Extensive amounts of layers also needs lots of computational power in training and prediction 

- It wasn't feasible until the raise of general-purpose graphic processing unit (GPGPU) computing 

- New techniques have been developed to speed up the training of DNNs and/or to avoid overfitting: [dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)), [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization), stochastic pooling, etc. They are used for other DNNs as well. 

# Deep learning: automated feature extraction

-   Conventional ML: manually craft a set of features in a process called **feature engineering**. 

-   Limitation: Sometimes features are too difficult to be manually
    designed, e.g., from I/O system log.

-   A (no-brainer) solution: let computers find it for us, even by
    brutal force.

-   Many DL tasks are end-to-end and hence are more black-box.


# Deep learning introduces new ways to use ANNs

-   There are more tasks that need function fitting beyond
    conventional classification and regression.

-   DL can be used to generate data (e.g, generating the translation of a sentence, turning a picture into Van Gogh style)

-   Sometimes we use the network to get something else useful, such
    as word embedding, but discard the network in the end.

-   Maybe weights of only a small set of layers are what we need
    from training.

-   Using DL requires creative ways to prepare training data (e.g., negative sampling), not  straightforward pairs of feature/input vectors and labels/output vectors).


# CNN, ConvNet, Convolutional Neural Networks

::: {.columns}
:::: {.column width=0.5}
-   Convolutional layers: 
    * [Convolution](https://en.wikipedia.org/wiki/Convolution) is a math operation that characterizes a signal (text, image, audio) using its spatial similarity with a template (called a **filter** in CNNs)
    * A CNN usually trains more than one filters, forming a **filter bank**. 
    * Conv layers are NOT fully connected. 

-   Pooling layers: downsample a matrix into one sample which contains the most useful information

-   ReLU layers: as activations, faster to train than $\tanh$ or logistic. 

-   Fully connected layer or dense layers (basically this is the regular ANN): usually toward  the end of a network. 

-   Softmax layers: common but not mandatory for output. 

::::

:::: {.column width=0.5}

-   Usual architecture: seveveral stacks of {conv, pool, Relu}, then several FC layers, finally/optionally softmax. 

-   Visualization of the output at layers: http://cs231n.github.io/convolutional-networks/

-   Too many hyperparameters (especially the size of filters and the stacking of layers). So some pre-configued architectures are commonly used, e.g., [AlexNet](https://en.wikipedia.org/wiki/AlexNet), VGGNet, [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network). 

-   Applications: matrix-like data, images, audios, 3D scans, time series, etc. 

-   Filters trained for a task can be reused or fine-tuned for another task. (**transfer learning**)
::::
:::

# Recurrent Neural Networks (RNNs)
::: {.columns}
:::: {.column width=0.6}

- Very often, the output of a system that we want to model depends partially on the previous output. E.g., next-word prediction or machine translation. 

- RNNs is the kind of networks. "recurrent" means using the same information again. 

- An RNN's input concatenated from two parts: $\mathbf{x}_t$, the "fresh" input at step $t$, and $\mathbf{o}_{t-1}$ the output at previous step $t-1$. They are transformed to the output $\mathbf{o}_t$ via two respective transfer matrixes: $\mathbf{o}_t=\mathbb{U}\mathbf{x}_t + \mathbf{W}\mathbf{o}_{t-1}$ 

- Unrolling/unfolding an RNN unit:
  ![a web image](http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg)

::::
:::: {.column width=0.4}
 
- (Many papers use $\mathbf{h}$ for $\mathbf{o}$ in the equation/figure left).

- Examples: Elman network [1](https://web.stanford.edu/group/pdplab/pdphandbook/handbookch8.html) [2](http://mnemstudio.org/neural-networks-elman.htm), [Hopfiled Network](https://en.wikipedia.org/wiki/Hopfield_network)

- An RNN is just an IIR filter (are you also a EE major?):
    $$y[n] = \sum_{l=1}^N a_l y[n-l] + \sum_{k=0}^{M} b_k x[n-k]$$ where
    $x[i]$ (or $y[i]$) is the $i$-th element (a scalar) in the input (or
    output) sequence.
::::
:::

# LTSM and GRU
<!-- ::: {.columns}
:::: {.column width=0.6} -->
- RNNs are not good at modeling long-term dependencies which is especially common in text. E.g., in the sentence, "Unlike what he said, I did NOT eat the pizza", "Unlike" and "not", two words separated far, need to be jointly considered to infer whether the subject ate the pizza. 

- A solution is to maintain a state. E.g., when "unlike" is scanned, set a state, later use that state to double-negate "eat" with "not". Reset the state after "not" is scanned.

- Hence LSTM (an architecture) was invented. Another motiviation is that gradient vanishing is significant for a deep RNN. 

- Instead of layers, LSTM has **cells** or **units**. Each LSTM cell has 3 **gates**: forget gate, input gate, and output gate. The 3 gates are
        computed from current input $\mathbf{x}(t)$ and output from
        previous cell $\mathbf{s}(t-1)$. Then "make a choice" between
        using previous state $\mathbf{c}(t-1)$ and current input and use
        the choice and output gate to make the final output.

- GRU: simpler, just reset gate and update gate.

- <http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/>

- <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>

LSTM ![image](figures/LSTM_unrolled.png){width="\\textwidth"} Source:
Listen, Attend, and Walk: Neural Mapping of Navigational Instructions to
Action Sequences, Mei et al., AAAI-16

# Neural language model in Elman network

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

# Neural language models

-   Feedforward: "A neural Probabilistic Language Model", Beigio et al.,
    JMLR, 3:1137--1155, 2003

-   "Recurrent neural network based language model", Mikolov et al.,
    Interspeech 2010

-   Multiplicative RNN: "Generating Text with Recurrent Neural
    Networks", Sutskever, ICML 2011

# Word2Vec



# Seq-to-seq learning

-   Let's go one more level up.

-   Instead of predicting the next element in an input sequence, can we
    produce the entire output sequence from the input sequence?

-   There could be no overlap between the two sequences, e.g., from a
    Chinese sentence to a German sentence.

-   Two RNNs: encoder and decoder

-   "Learning Phrase Representations using RNN Encoder--Decoder for
    Statistical Machine Translation", Cho et al., EMNLP 2014

# Transformer

# Autoencoder 

# GAN 

# Transfer learning

