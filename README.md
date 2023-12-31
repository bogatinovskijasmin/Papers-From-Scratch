# Papers Implementations From Scratch

It contains dozens of implementations of some of the most interesting ideas in Machine Learning over the past two decades. Note that for the implementation of some of the methods, the class **torch.nn** is not used. This is due to the intention to practice the internal inner working of the commonly used core methods and ideas. This is a useful resource for anyone interested in learning the internals of the covered models and techniques. 
The papers are organized based on the topic they cover. The main motivation for this project is to practice some of the concepts that shape the field of predominantly Deep Learning.
Currently, there are **29** papers that are implemented. 

**Overview at a glance:**

1. **Sequence Modeling** 

    *1.1. World-level Language Methods*

    *1.2. Translation Methods*

    *1.3. Character-level Language Methods*

2. **Computer Vision**

    *2.1. Popular Vision Architectures*

    *2.2. Vision Applications*

3. **Other Topics**

    *3.1. Common Optimization Methods Used in Deep Learning*

    *3.2. Recommendation Methods*

    *3.3. General Adversarial Networks*

The considered datasets are the following: 
1. The book "Time Machine" by H. G. Wells [1898]
2. English to French bilingual translation pairs http://www.manythings.org/anki/
3. Penn Tree Bank (https://catalog.ldc.upenn.edu/LDC99T42)
4. Wiki-2 (preprint arXiv:1609.07843)
5. ImageNet 
6. Fashion MNIST (build-in in PyTorch)
7. Movies Lens (http://files.grouplens.org/datasets/movielens/ml-100k.zip)
8. Click through Rate (McMahan, H. B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., … others. (2013). Ad click prediction: a view from the trenches. Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1222–1230).)
9. Pokemondb (https://pokemondb.net/sprites)
10. Synthetic data created with built-in torch functions
11. Shakespeare (https://huggingface.co/datasets/tiny_shakespeare)

The groups of papers are the following:

## 1. Sequence Modeling

### 1.1.  Word Level Language Models
Contains implementation of several popular language models on word level. The current methods implemented are:

1. skip-gram: **Distributed representations of words and phrases and their compositionality.** Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Advances in neural information processing systems (pp. 3111–3119).
2. BERT: **Bert: pre-training of deep bidirectional transformers for language understanding.** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018).  arXiv preprint arXiv:1810.04805. 
3. GPT-3: **Language Models are Few-Shot Learners**, Sutskever, I., Amodei D. (2020) https://arxiv.org/abs/2005.14165

### 1.2. Translation Models
In this folder, several approaches for sequence-to-sequence (Seq2Seq) modeling are implemented. 

1. encoder_decoder_GRUs: **Sequence to sequence learning with neural networks.** Sutskever et al., 2014
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Advances in neural information processing systems (pp. 3104–3112).
2. encoder_decoder_addetive_attention: **Neural machine translation by jointly learning to align and translate.** Bahdanau et al., 2014
Bahdanau, D., Cho, K., & Bengio, Y. (2014). arXiv preprint arXiv:1409.0473.
3. encoder_decoder_multi-head_attention: The **Transformer architecture Attention is all you need.** Vaswani et al., 2017
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Polosukhin, I. (2017). Advances in neural information processing systems (pp. 5998–6008).


### 1.3. Character Level Language Models
A recurrent model uses a character-level representation of the words in the book "Time Machine" to learn the language model.
 It implements the papers listed below:

This project implements the following papers: 

1. LSTM **Long short-term memory.** Hochreiter, S., & Schmidhuber, J. (1997). Neural Computation, 9(8), 1735–1780.
2. GRU **On the properties of neural machine translation: encoder-decoder approaches.** Cho, K., Van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014).  arXiv preprint arXiv:1409.1259.
3. RNN **Learning internal representations by error propagation.** Rumelhart, David E; Hinton, Geoffrey E, and Williams, Ronald J (Sept. 1985). Tech. rep. ICS 8504. San Diego, California: Institute for Cognitive Science, University of California.
4. Stacked RNN **Learning internal representations by error propagation.** Rumelhart, David E; Hinton, Geoffrey E, and Williams, Ronald J (Sept. 1985). Tech. rep. ICS 8504. San Diego, California: Institute for Cognitive Science, University of California.
5. Bidirectional RNN **Bidirectional recurrent neural networks.** Schuster, M., & Paliwal, K. K. (1997). IEEE Transactions on Signal Processing, 45(11), 2673–2681.


## 2. Computer Vision

### 2.1. Popular Vision Architectures
This project implements the following popular vision architectures: 

1. LeNet: **Gradient-based learning applied to document recognition.**, LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., & others. (1998). Proceedings of the IEEE, 86(11), 2278–2324.
2. AlexNet: **Imagenet classification with deep convolutional neural networks.**, Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Advances in neural information processing systems (pp. 1097–1105).
3. VGG: **Very deep convolutional networks for large-scale image recognition.**, Simonyan, K., & Zisserman, A. (2014).  arXiv preprint arXiv:1409.1556.
4. GoogleNet: **Going deeper with convolutions.**, Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … Rabinovich, A. (2015). Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1–9)
5. ResNet: **Deep residual learning for image recognition.**, He, K., Zhang, X., Ren, S., & Sun, J. (2016). Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770–778)
6. NiNet: **Network in network.**, Lin, M., Chen, Q., & Yan, S. (2013). arXiv preprint arXiv:1312.4400.
7. AnyNet (RegNetX): **Designing network design spaces.**, Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10428–10436).


### 2.2. Vision Applications:
This project implements the following paper:

1. Style Transfer: **Image style transfer using convolutional neural networks.**,  Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414–2423).

## 3. Other Topics:

### 3.1. Common Optimization Methods used in Deep Learning
This project closely follows the following literature:

1. Momentum: **Lectures on convex optimization.** Nesterov, Y. (2018).  Vol. 137. Springer. 
2. AdaGrad: **Adaptive subgradient methods for online learning and stochastic optimization.** Duchi, J., Hazan, E., & Singer, Y. (2011). Journal of Machine Learning Research, 12(Jul), 2121–2159.
3. RMSProp: **Lecture 6.5-rmsprop: divide the gradient by a running average of its recent magnitude.** Tieleman & Hinton, 2012
Tieleman, T., & Hinton, G. (2012). COURSERA: Neural networks for machine learning, 4(2), 26–31.
4. Adadelta: **An adaptive learning rate method.** arXiv preprint arXiv:1212.5701. Zeiler, M. D. (2012).
5. Adam: **A method for stochastic optimization.** arXiv preprint arXiv:1412.6980. Kingma, D. P., & Ba, J. (2014).

### 3.2. Recommendation Methods
This project implements the following approaches:
1. Matrix Factorization; **Matrix factorization techniques for recommender systems.** Koren, Y., Bell, R., & Volinsky, C. (2009). Computer, pp. 30–37.
2. AutoRec: **Autorec: autoencoders meet collaborative filtering.**  Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015). Proceedings of the 24th International Conference on World Wide Web (pp. 111–112).
3. Factorization Machines; **Factorization machines.** Rendle, S. (2010). 2010 IEEE International Conference on Data Mining (pp. 995–1000).
4. Deep Factorization Machines (DeepFM); **Deepfm: a factorization-machine based neural network for ctr prediction.**  Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). Proceedings of the 26th International Joint Conference on Artificial Intelligence (pp. 1725–1731).

### 3.3. General Adversarial Networks
1. DCGAN: **Unsupervised representation learning with deep convolutional generative adversarial networks.**, Radford, A., Metz, L., & Chintala, S. (2015). arXiv preprint arXiv:1511.06434.

While the code is written by me, on places for the sake of convinence, it closely follows code implementations from the book "Dive Into Deep Learning" Smola, A. et al. (e.g., for data downloading, loading etc.), licensed under https://github.com/d2l-ai/d2l-en/blob/master/LICENSE.