---
title:  Things to pay attention to after Lecture 4 - Linear Regression
date: 2019-02-14
author: Pantelis Monogioudis
tags:
  [
    "cs634",
  ]
excerpt: This is a short list of things to pay attention with respect to Linear Regression. 
draft: false
cover: ./images/regression.png
publishedAt: here.
canonicalLink:
image: ./images/regression.png
avatar: avatars/lekoarts.png
imageAuthor: 
imageAuthorLink: 
imageTitle: OLS
showImageInArticle: true
---

In the lecture a single-dimensional data set with just 10 data points and a non-linear relationship between x and y was given. Despite its simplicity this synthetic problem is very instructive. You need to:

1. Be able to distinguish what is a non-linear problem and what is a linear problem. 
2. Be familiar with the ML problem statement - both in terms of terminology and the probability distributions involved. 
3. Write down the parametric form of the model e.g. $\mathbf{w}^T \phi(x)$ and understand how this enters the Loss function. 
4. Understand the MSE behavior with number of datapoints, parametric model complexity (capacity) and what can be done to reduce generalization error due to overfitting (regularization).
5. Understand the principle behind Byeasian linear regression especially how it can help you in problems where you have unequal data coverage (areas where there are lots of data and areas that are small number of data points).
6. Finally in terms of memorizing math, you need to be able to write down the parametric model equation, the MSE and the Bayes rule only.