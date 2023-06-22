# Word-level Language Models

In this project several language models on word level are trained. The "skip-gram" model is context independent, while BERT and GPT are context dependent. 
The "context"-dependacy refers to the fact wether the embedding of a certain word depends on the context in which the word is present. 

1. skip-gram: **Distributed representations of words and phrases and their compositionality.** Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Advances in neural information processing systems (pp. 3111–3119).
2. BERT: **Bert: pre-training of deep bidirectional transformers for language understanding.** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018).  arXiv preprint arXiv:1810.04805.

As a dataset used a sample of Wikipedia articles. The data preparation scripts were obtained from the book "Dive Into Deep Learning". 