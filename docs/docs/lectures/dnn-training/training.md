---
title: Training DNNs
---

## Training a single neuron - the weight space
For convenience let us study the case where the input vector $\mathbf x$ and the parameter vector $\mathbf w$ are both two-dimensional and the function performed by the neuron is: 

$h = g(\mathbf x;\mathbf w) = \frac{1}{1+e^{-(w_1 x_1 + w_2 x_2)}} $

![](images/two-weight-single-neuron.png)


Each  point in $\mathbf w$ space corresponds to a function of $\mathbf x$. 

![](images/weight-space.png)

We have seen the cross-entropy loss function. Its derivative $\mathbf g = \partial L/\partial \mathbf w$ is given by:

$ \frac{\partial L}{\partial w_j} =  \sum_i - (y^{(i)} - \hat{y}^{(i)}) x_j^{(i)} $
    
Notice the quantity $e^{(i)} \equiv y^{(i)} - \hat{y}^{(i)}$ -  it tells us that the rate at which the neuron learns depends on the error in the output. The larger the error, the faster the neuron will learn. The simplest thing to do with a gradient of a cost function is to descend it. Since the derivative  $\partial L/\partial \mathbf w$  is a sum of terms $\nabla^{(i)}$ defined by:

$ \nabla J_j^{(i)} \equiv - (y^{(i)} - \hat{y}^{(i)}) x_j^{(i)} $
    
for $i=1,\ldots, {m'}$, we can obtain a simple on-line algorithm by putting each input through the network one at a time, and adjusting $\mathbf w$ a little (learning rate $\epsilon$) in a direction opposite to $\nabla^{(i)}$.


![playground-single-neuron](images/playground-single-neuron.png)

## Depth and Performance of DNNs. 

![](images/accuracy-vs-depth.png)

### Regularization

![](images/reg_strengths_cs231n.jpeg)
   
Some use cases in tensorflow playground require to address overfitting by adding in the objective function some tunable penalty term that prevents. Such penalty term is usually:

$\lambda J_{penalty} = \lambda \left(\sum_l W_{(l)}^2 \right) $

where $l$ is the hidden layer index and $W$ is the weight tensor. 
    
