### Popular Vision Architectures
Most of the considered Vision Architectures address the problem of image classification. They make use of the inductive bias available through convolutional neural networks. The selection of the networks
is mostly based on the novel components they introduced, which motivated many modern computer vision systems. 

For example, LeNet introduced the application of the ConvNet operator on image classification. AlexNet enabled to train of deeper networks
by introducing GPUs for parallel training, MaxPool, dropout, ReLU and similar other tricks to enable efficient training of the network.
VGG introduced the idea of using blocks of several ConvLayers prior to a Pooling layer to learn more complex representations. 
GoogleNet introduced the **Inception block** that eliminates the problem of selecting the kernel size as it introduced a block
composed of a multitude of those kernels (i.e., it concatenates representations of 1x1, 3x3 and 5x5 kernels). This resulted in the ability to train smaller and deeper networks with improved performance. ResNet introduced the idea of enriching the expressiveness
of the family function by ensuring that as more layers are added, the expressive power of the resulting network is at least as 
rich as the smaller one. This is done by learning f(x) = x, i.e., propagating the input information towards the output through a residual layer. Note that it also uses learning blocks where the block is implementing the idea 
of grouped convolutions (i.e., multiple view split of the different channels, combined at the end of the block). 
NiNet introduces the idea of average pooling of the last layer with 1x1 Conv layers within each block, so the head doesn't suffer from the high dimensionality of a dense layer at the top (e.g., VGG uses up to 400MB storage space for the last layer on Imagenet). 
AnyNet introduces the idea of systematically building a deep learning architecture by carefully stacking different building blocks in a stem-body-head architecture. As different building blocks are used, the blocks are introduced in previous works. One popular block is the RegNeXt block which introduces a block similar to the Inception, with added residual connections and sandwiched conv-layers of size 3x3 in-between two 1x1 conv layers. The joint paper with AnyNet stores a rich set of guidelines on how to design modern networks. 

The major architectural style prior to the rise of the Transformers has been the idea of composing the network from
**a. Stem:** to learn high-level features, **b. Body:** to learn a complex representation of the spatial structure within the image and **c. Head:** used to perform a certain task, e.g., image classification.
A body is typically composed of multiple stages. Each stage is composed of multiple blocks. The first block in each stage is usually compressing the input by a factor of 2 (stride 2). The rest of the blocks within the stage are learning different output representations. 

Alongside the following notes, this project implements the following popular vision architectures: 

1. LeNet: **Gradient-based learning applied to document recognition.**, LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., & others. (1998). Proceedings of the IEEE, 86(11), 2278–2324.
2. AlexNet: **Imagenet classification with deep convolutional neural networks.**, Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Advances in neural information processing systems (pp. 1097–1105).
3. VGG: **Very deep convolutional networks for large-scale image recognition.**, Simonyan, K., & Zisserman, A. (2014). arXiv preprint arXiv:1409.1556.
4. GoogleNet: **Going deeper with convolutions.**, Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … Rabinovich, A. (2015). Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1–9)
5. ResNet: **Deep residual learning for image recognition.**, He, K., Zhang, X., Ren, S., & Sun, J. (2016). Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770–778)
6. NiNet: **Network in network.**, Lin, M., Chen, Q., & Yan, S. (2013). arXiv preprint arXiv:1312.4400.
7. AnyNet (RegNetX): **Designing network design spaces.**, Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10428–10436).
8. convolutional_operators.py implements the convolutional and pooling layers from scratch. 