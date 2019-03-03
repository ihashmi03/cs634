---
title:  Things to pay attention to after Lecture 3 - Math Review 
date: 2019-02-07
author: Pantelis Monogioudis
tags:
  [
    "cs634",
  ]
excerpt: This is a short list of things to pay attention to after the Math Review. 
draft: false
cover: ./images/antoine-dautry-428776-unsplash.jpg
publishedAt: here.
canonicalLink:
image: ./images/antoine-dautry-428776-unsplash.jpg
avatar: avatars/lekoarts.png
imageAuthor: Antoine Dautry
imageAuthorLink: https://unsplash.com/@clemhlrdt
imageTitle: Math
showImageInArticle: true
---

The math review spanned many topics. 

On probability theory you need to understand: 

* The conditional probability and the Bayes theorem. Given a problem statement (e.g. Cancer diagnosis) you need to be able to, using the Bayes rule, to derive conditional probabilities. The Bayes rule is something you need to be able to memorize. The way I had is to write the posterior probability p(a|b) equal to the flipped conditional p(b|a) multiplied by a fraction. Imagine that a|b is a fraction a/b and therefore it is p(a)/p(b) - the whole thing then becomes p(a|b)=p(b|a) p(a)/p(b).
* The bivariate Gaussian and binomial distributions without trying to memorize the formulas - the level of understanding requires for Gaussian is to be able to tell the difference between positive and negative correlated data. 

On linear algebra you will need:  

* The principle of Singular Value Decomposition without memorizing the formula. You need to get the intuition behind selecting the k stringest eigenvalues and what we achieve by doing so. 

On optimization you will need:

* Batch and Stochastic gradient descent. You need to get the intuition as to why the algorithm can converge and what is the role of the learning rate.