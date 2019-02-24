---
title: Logistic Regression
---

Logistic regression is used in machine learning extensively - every time we need to provide probabilistic semantics to an outcome e.g. predicting the risk of developing a given disease (e.g. diabetes; coronary heart disease), based on observed characteristics of the patient (age, sex, body mass index, results of various blood tests, etc.), whether an voter will vote for a given party, predicting the probability of failure of a given process, system or product, predicting a customer's propensity to purchase a product or halt a subscription, predicting the likelihood of a homeowner defaulting on a mortgage. Conditional random fields, an extension of logistic regression to sequential data, are used in Natural Language Processing (NLP). The common denominator in all above cases is that the outcome is an assignment to a distinct class.

## Binary case
If we consider the two class problem, we can write the posterior probability as,

$$p(\mathcal{C}_1|\mathbf{x}) = \frac{p(\mathbf{x}|\mathcal{C}_1) p(\mathcal{C}_1)}{p(\mathbf{x}|\mathcal{C}_1) p(\mathcal{C}_1) + p(\mathbf{x}|\mathcal{C}_2) p(\mathcal{C}_2)} = \frac{1}{1 + exp(-\alpha)} = \sigma(\alpha)$$

where $\alpha = \ln \frac{p(\mathbf{x}|\mathcal{C}_1) p(\mathcal{C}_1)}{p(\mathbf{x}|\mathcal{C}_2) p(\mathcal{C}_2)}$

Assuming that the class-conditional densities $p(\mathbf{x}|\mathcal{C}_1)$ and $p(\mathbf{x}|\mathcal{C}_2)$ are Gaussian, the posterior distribution is given by

$$p(\mathcal{C}_1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + w_0)$$

where the weights are the closed form solutions ($w_0$ is omitted),

$$\mathbf{w} = \Sigma^{-1}(\mu_1 - \mu_2)$$

The parameters $\mu$ and $\Sigma$ can be estimated using Maximum Likelihood. 

The figure below shows the corresponding posterior distribution $p(\mathcal{C}_1|\mathbf{x})$

![posterior-two-class-example](images/Figure4.10a.png)
*The class-conditional densities for two classes, denoted red and blue.*

![posterior-two-class-example](images/Figure4.10b.png)
*On the right is the corresponding posterior probability for the red class, which is given by a logistic sigmoid of a linear function of $x$.*

With logistic regression we skip the assumption about the class-conditional densities as they add parameters to our problem that grow  quadratic to the number of dimensions and we attempt to find the $n$ parameters of the model directly (the number of features) and sure enough we will use ML to do so. 

By repeating the classical steps in ML methodology i.e. writing down the expression of the likelihood function (this will now be a product of binomials), we can write down the negative log likelihood function as, 

$$L(\mathbf{w}) = - \ln p(\mathbf{y},\mathbf{w}) = - \sum_{i=1}^m \{y_i \ln \hat{y}_i + (1-y_i) \ln (1-\hat{y}_i) \}$$
 
which is called **cross entropy error function** - probably the most widely used error function in classification due to its advantages such as its probabilistic and information theoretic roots as well as its shape shown in the figure below. 

![cross-entropy](images/cross-entropy-binary.png)

Minimizing the error function with respect to $\mathbf{w}$ by taking its gradient 

$$\nabla L_{\mathbf{w}} = \sum_{i=1}^m (\hat{y}_i - y_i) x_i$$

that defines the batch gradient decent algorithm. We can then readily convert this algorithm to SGD by considering mini-batch updates.
