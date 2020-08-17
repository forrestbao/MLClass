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
- We are incompetent. No kidding! Sometimes it is very difficult to come up with step-by-step instructions. Examples: 
    0. Self-driving
    1. AlphaGo 
    2. Automated circuit routing
    3. Machine translation
    4. Commonsense reasoning, text entailment 
    5. Document generation, auto-reply of messages/emails
    6. Next word predition
- It is a dream. "Creating an artificial being has been the dream since the beginning of science." -- Movie A.I., Spielberg et al., 2001
- \$\$\$! 

# Three types of MLs

ML is all about finding a function. 

- Supervised, finding $\hat{f}(x) \approx f(x)$ with ground truth provided by human. 
    * Let $x$ and $y$ be two (vectors of) variables. The function $f$ is the relation between $x$ and $y$. But only god knows $f$. 
    * We construct another function $\hat{f}$ to approximate $f$ such that $\hat{y} = \hat{f}(x) \approx y = f(x)$ for a given $x$. 
    * **Supervised** because we  provide many pairs of $x$'s and $y$'s for the computer to know the difference between $\hat{y}$ and $y$ on a large pool of samples. 
    * Examples: Determine whether a picture contains a cat, [Flavia](http://flavia.sourceforge.net/), [determine who have contracted COVID-19 from blood profile](https://arxiv.org/abs/2005.06546), determine the gesture command (e.g., scroll up, zoom in, etc.) from camera data.
    * More than categorization/classification: [Mflux](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004838), [Review helpfulness prediction](https://www.aclweb.org/anthology/P15-2007.pdf), [Document summarization](https://www.aclweb.org/anthology/E17-2112.pdf), [I/O stream prediction]
- Unsupervised, finding $\hat{f}(x)$ without ground truth
- Reinforcement, let the machine find ground truth itself

# Representation of $x$ (and $y$)

- $x$ is usually not a simple (vector of) number(s). How to tell it to a computer? 
- Feature engineering vs. deep learning

