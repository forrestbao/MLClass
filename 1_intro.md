---
title: | 
         CS 474/574 Machine Learning \
         1. Introduction
author: |
          Prof. Dr. Forrest Sheng Bao \
          Dept. of Computer Science \
          Iowa State University \
          Ames, IA, USA \
date:   \today
header-includes: |
     \usepackage{amssymb}
     \usefonttheme[onlymath]{serif}
---

# Why Machine Learning (ML)

- How computers know how to do things?
- Two ways:
    1. programming: steps detailed by human programmer
    2. learning: without being specifically told 
- Example 1: machine translation
    1. programming: writing many rules to replace and reposition words, e.g., 
    _Do you speak Julia?_ _Sprechen Sie Julia?_   
    2. learning: feeding the computer many bilingual documents 
- Example 2: sorting 
    1. programming: Quicksort, etc. 
    2. learning: feeding the computer many pairs of unsorted and sorted list of numbers. 
- The first approach in the context of AI is also called rule-based system or expert system, e.g. MyCin, Grammarly. 

# Why ML is attractive

- We are lazy. We want to shift the heavy lifting to the computers. 
- We are incompetent. No kidding! Sometimes it is very difficult to come up with step-by-step instructions. 
- Examples: Self-driving, AlphaGo, Automated circuit routing, Machine translation, Commonsense reasoning, text entailment, Document generation, auto-reply of messages/emails, [fly a helicoper inversely](https://www.youtube.com/watch?v=M-QUkgk3HyE), [van-Gogh-lize paints](https://blogs.nvidia.com/blog/2016/05/25/deep-learning-paints-videos/). 
- It is a dream. "Creating an artificial being has been the dream since the beginning of science." -- Movie A.I., Spielberg et al., 2001

# Three types of MLs

ML (in current approaches) is about finding/approximating functions. 

- Supervised, finding $\hat{f}(x) \approx f(x)$ with ground truth provided by human. 
    * Let $x$ and $y$ be two (vectors of) variables. The function $f$ is the relation between $x$ and $y$. But only god knows $f$. 
    * We construct another function $\hat{f}$ to approximate $f$ such that $\hat{y} = \hat{f}(x) \approx y = f(x)$ for a given $x$. 
    * **Supervised** because we  provide many pairs of $x$'s and $y$'s for the computer to know the difference between $\hat{y}$ and $y$ on a large pool of samples. 
    * Examples: object detection from images, [Flavia](http://flavia.sourceforge.net/), [CPU branch prediction](https://www.electronicdesign.com/technologies/microprocessors/article/21802106/ai-helps-amds-ryzen-take-on-intel),  [COVID-19 diagnosis from blood profile](https://arxiv.org/abs/2005.06546), [Epileptic EEG recognition](https://www.technologyreview.com/2009/04/29/213440/a-neural-net-that-diagnoses-epilepsy/), [depression treatment from brain shapes](https://mfr.osf.io/render?url=https://osf.io/b58jr/?action=download%26mode=render).
    * Beyond categorization/classification: [Mflux](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004838), [Review helpfulness prediction](https://www.aclweb.org/anthology/P15-2007.pdf), [Document summarization](https://www.aclweb.org/anthology/E17-2112.pdf), predict house prices
- Unsupervised, finding $\hat{f}(x)$ without ground truth
- Reinforcement, let the machine find ground truth itself

# Representation of $x$

- $x$ is usually not a simple (vector of) number(s). How to tell it to a computer? 
- Example: bananas vs. apples
- **Feature engineering**: manually craft functions to **extract** features from raw data, e.g,. SIFT, bag-of-words. 
- Automated feature extraction in deep learing: E.g., filters in CNNs. 
- If $x$ involves categorical values (e.g., gender), there are usually two approaches: [**One-hot encoding**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) and [**embedding**]() (in DL context, to be discussed later). 

# Supervised ML
- Given many pairs of $x$'s and $y$'s such that each $y$ is the output of a function $f: \mathbb{R}^m \mapsto \mathbb{R}^n$ for a corresponding input $x$ (i.e., $y=f(x)$), construct a function $\hat{f}$ that approximates $f$. 
- By "approximate", we usually mean to minimize $||\hat{f}(x) - f(x)||^{p}$ where $p$ is usually 1 or 2. [See $\ell_p$-norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) . 
- In other words, $f$ is a black box. And we need to find $\hat{f}$ that mimick the black box. 
- $x$ is called the **input** (especially when raw data is used as the input) or **feature vector** (if using feature engineering). $y$ is called the **label** (in classification) or **target** (used more often recently). 
- Classification vs. Regression: When $y$ is continuous or discrete. 
- In modern DL context, such division is usually no mentioned, expecially in generative tasks. 
