### Generative Adversarial Networks

Supervised learning tasks are one of the most popular tasks in machine learning. 
They belong to the class of discriminative models. A parallel class of models are  
generative models that have the capability of reconstructing properties of the input data distribution.
One of the most popular classes of these models is the General Adversarial Networks (GANs for short). 
The key idea in GAN is:

1. Start with data from input distribution (your original training data);
2. Create one network referred to as a discriminator;
3. Create a second network referred to as a generator; The output of the generator has the same dimensional shape as the size of the input (from the training data).
4. Pass the training sample data from the target data through the discriminator and label the sample as positive;
5. Calculate the part of the loss associated with the original training data. 
5. Sample a data from known random distribution, e.g., Gaussian (batch size, latent_dim, 1, 1);
6. Pass the sampled data through the generator and its output through the discriminator. 
7. Use the output of the discriminator to calculate the second part of the loss of the two networks. Label the sampled data as negative.
8. The total loss for training the discriminator is now complete, and it is used to update the parameters of the discriminator.
9. The output of the generator is then fed into a separate loss function. The ground truth for the loss, in this case, is a positive class. (opposite of the previous point).
10. As a loss function, a possible choice is Binary Cross Entropy (commonly used with logits).

The two networks use separate optimizers for updating the parameters.   

This project implements the following paper:

1. DCGAN: **Unsupervised representation learning with deep convolutional generative adversarial networks.**, Radford, A., Metz, L., & Chintala, S. (2015). arXiv preprint arXiv:1511.06434.

The implementation of DCGAN closely follows the implementation of "Dive Into Deep Learning" by Smolla et al.