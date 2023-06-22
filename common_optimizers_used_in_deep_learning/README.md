# Common Optimization Techniques in Deep Learning (Algorithms and Scheduling strategies) 

Given a task, data, model in the form of a neural network, and a loss that evaluates how well the model fits the data, 
the goal of the optimization algorithms is to adjust the parameters of the network such that the model best fits the data. 

The gradient-based optimization algorithms provide a way how to adjust the parameters of the network to minimize the training loss. 
Therefore, they are a key component. As the optimization functions in deep learning are commonly highly un-linear, and it is 
challenging (and even not desirable) to attain the global optimum, they are of essential value. In addition, by optimizing their parameters
(e.g., by scheduling the learning rate) one can adjust the training procedure ensuring that the parameters of the network are
changing accordingly. This is essential to ensure that the network is learning (e.g., no dead neurons or layers exist). 

This project implements the following commonly used methods from scratch. Note that PyTorch is used as a tool to provide automatic differentiation
capabilities, but the methods are implemented as separate modules (not taken from the torch.optim module of PyTorch).

This project closely follows the following literature: 

1. Momentum: **Lectures on convex optimization.** Nesterov, Y. (2018).  Vol. 137. Springer. 
2. AdaGrad: **Adaptive subgradient methods for online learning and stochastic optimization.** Duchi, J., Hazan, E., & Singer, Y. (2011). Journal of Machine Learning Research, 12(Jul), 2121–2159.
3. RMSProp: **Lecture 6.5-rmsprop: divide the gradient by a running average of its recent magnitude.** Tieleman & Hinton, 2012
Tieleman, T., & Hinton, G. (2012). COURSERA: Neural networks for machine learning, 4(2), 26–31.
4. Adadelta: **An adaptive learning rate method.** arXiv preprint arXiv:1212.5701. Zeiler, M. D. (2012).
5. Adam: **A method for stochastic optimization.** arXiv preprint arXiv:1412.6980. Kingma, D. P., & Ba, J. (2014).

In all the projects the FashionMNIST dataset and the softmax regression model are used.