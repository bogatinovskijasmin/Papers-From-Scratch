# Papers From Scratch

It contains dozens of my own implementations of dozens of papers. The papers are organized based on the topic they cover. 
The main motivation for this project is to practice some of the concepts that shape the field of predominantly Deep Learning.
Currently, the groups of papers are the following:

### seq2seq_models
In this folder, several approaches for sequence-to-sequence (Seq2Seq) modeling are implemented. 

1. encoder_decoder_GRUs: **Sequence to sequence learning with neural networks.** Sutskever et al., 2014
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Advances in neural information processing systems (pp. 3104–3112).
2. encoder_decoder_addetive_attention: **Neural machine translation by jointly learning to align and translate.** Bahdanau et al., 2014
Bahdanau, D., Cho, K., & Bengio, Y. (2014). arXiv preprint arXiv:1409.0473.
3. encoder_decoder_multi-head_attention: The **Transformer architecture Attention is all you need.** Vaswani et al., 2017
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Polosukhin, I. (2017). Advances in neural information processing systems (pp. 5998–6008).


### character_level_language_models
A recurrent model uses a character-level representation of the words in the book "Time Machine" to learn the language model.
Note that for the implementation the default models are from the torch.nn package is not used. It implements the papers listed below:

This project implements the following papers: 

1. LSTM **Long short-term memory.** Hochreiter, S., & Schmidhuber, J. (1997). Neural Computation, 9(8), 1735–1780.
2. GRU **On the properties of neural machine translation: encoder-decoder approaches.** Cho, K., Van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014).  arXiv preprint arXiv:1409.1259.
3. RNN **Learning internal representations by error propagation.** Rumelhart, David E; Hinton, Geoffrey E, and Williams, Ronald J (Sept. 1985). Tech. rep. ICS 8504. San Diego, California: Institute for Cognitive Science, University of California.
4. Stacked RNN **Learning internal representations by error propagation.** Rumelhart, David E; Hinton, Geoffrey E, and Williams, Ronald J (Sept. 1985). Tech. rep. ICS 8504. San Diego, California: Institute for Cognitive Science, University of California.
5. Bidirectional RNN **Bidirectional recurrent neural networks.** Schuster, M., & Paliwal, K. K. (1997). IEEE Transactions on Signal Processing, 45(11), 2673–2681.


 
### common_optimizers_used_in_deep_learning

This project closely follows the following literature: 

1. Momentum: **Lectures on convex optimization.** Nesterov, Y. (2018).  Vol. 137. Springer. 
2. AdaGrad: **Adaptive subgradient methods for online learning and stochastic optimization.** Duchi, J., Hazan, E., & Singer, Y. (2011). Journal of Machine Learning Research, 12(Jul), 2121–2159.
3. RMSProp: **Lecture 6.5-rmsprop: divide the gradient by a running average of its recent magnitude.** Tieleman & Hinton, 2012
Tieleman, T., & Hinton, G. (2012). COURSERA: Neural networks for machine learning, 4(2), 26–31.
4. Adadelta: **An adaptive learning rate method.** arXiv preprint arXiv:1212.5701. Zeiler, M. D. (2012).
5. Adam: **A method for stochastic optimization.** arXiv preprint arXiv:1412.6980. Kingma, D. P., & Ba, J. (2014).