import collections
import random
import re
import torch
from d2l import torch as d2l


class TimeMachine(d2l.DataModule):
    def _download(self):
        fname = d2l.download(d2l.DATA_URL+'timemachine.txt', self.root, '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', " ", text).lower()

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_text, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))[:100000]
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        super().__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val

        corpus, self.vocab = self.build(self._download())
        array = torch.tensor([corpus[i:i+num_steps+1] for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:, :-1], array[:, 1:]

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, self.num_train+self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

    def get_tensorloader(self, tensors, train, idx):
        tensors = [a[idx] for a in tensors]
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset,
                                           shuffle=train,
                                           batch_size=self.batch_size)


class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [token for token, freq in self.token_freqs if freq >=min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices)>1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    @property
    def unk(self):
        return self.token_to_idx['<unk>']





# data = TimeMachine()
# raw_text = data._download()
# data = TimeMachine()
# corpus, vocab = data.build(raw_text)



# tokens = data._tokenize(text)
# print(','.join(tokens[:30]))
#
# vocab = Vocab(tokens)
# indices = vocab[tokens[:10]]

# text = data._preprocess(raw_text)
# words = text.split()
# word_vocab = Vocab(words)
#
#
# print("ZIPF Law for token distribution")
# freqs = [freq for token, freq in word_vocab.token_freqs]
# d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
#          xscale='log', yscale='log')
#
# bigrams = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
# bigrams_vocab = Vocab(bigrams)
# print(bigrams_vocab.token_freqs[:10])
#
# trigrams = ['--'.join(triplets) for triplets in zip(words[:-2], words[1:-1], words[2:])]
# trigrams_vocab = Vocab(trigrams)
#
# bigram_freqs = [freq for token, freq in bigrams_vocab.token_freqs]
# trigram_freqs = [freq for token, freq in trigrams_vocab.token_freqs]
# d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
#          ylabel='frequency: n(x)', xscale='log', yscale='log',
#          legend=['unigram', 'bigram', 'trigram'])
