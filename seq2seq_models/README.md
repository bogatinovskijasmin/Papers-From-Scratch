# Sequence-to-Sequence Models

In practice it is common to have problems with arbitrary lengths of input and output. For example, the problem of translation between two 
languages can be seen as mapping the input sequence of the source language to a variable length sequence of the target language. Similar problems
are mapping between text and sound, sequence and shifted version of the sequence among others. 

In this folder several approaches for sequence-to-sequence (Seq2Seq) modeling are implemented. As of this point it supports methods proposed in the following papers:

1. encoder_decoder_GRUs: *Sequence to sequence learning with neural networks.* Sutskever et al., 2014
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Advances in neural information processing systems (pp. 3104–3112).
2. encoder_decoder_addetive_attention: *Neural machine translation by jointly learning to align and translate.* Bahdanau et al., 2014
Bahdanau, D., Cho, K., & Bengio, Y. (2014). arXiv preprint arXiv:1409.0473.
3. encoder_decoder_multi-head_attention: The **Transformer** architecture *Attention is all you need.* Vaswani et al., 2017
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Polosukhin, I. (2017). Advances in neural information processing systems (pp. 5998–6008).

   
As a dataset used a sample of sentance pairs for English (as source language) and French (as target language). The data preparation 
scripts were obtained from the book "Dive Into Deep Learning". 