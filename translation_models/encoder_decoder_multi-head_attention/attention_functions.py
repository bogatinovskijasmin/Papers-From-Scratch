import math
import torch



def masked_softmax(features, valid_seq_lens):
    # features is 3D matrix
    # valid_seq_len is 1D or 2D matrix
    def _mask_seq(X, valid_seq_len, value=0):
        x_len = X.size(1)
        mask = torch.arange((x_len), device=X.device, dtype=torch.int32)[None, :] < valid_seq_len[:, None]
        print(X.shape, mask.shape)
        X[~mask] = value
        return  X

    if valid_seq_lens is None:
        return torch.nn.functional.softmax(features, dim=-1)
    else:
        shape = features.shape
        if valid_seq_lens.dim() == 1:
            valid_seq_lens = torch.repeat_interleave(valid_seq_lens, shape[1])
            # print("1. valid_seq_lens.shape", valid_seq_lens.shape)
        else:
            valid_seq_lens = valid_seq_lens.reshape(-1)
        rescaled_features = _mask_seq(features.reshape(-1, shape[-1]), valid_seq_len=valid_seq_lens, value=-1e6)
        return torch.nn.functional.softmax(rescaled_features.reshape(shape), dim=-1)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dropout,num_hiddens, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.W_q = torch.nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = torch.nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = torch.nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = torch.nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_seq_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        if valid_seq_lens is not None:
            valid_seq_lens = torch.repeat_interleave(valid_seq_lens, repeats=self.num_heads, dim=0)
        output = self.attention.forward(queries, keys, values, valid_seq_lens)
        output = self.transponse_output(output)
        output = self.W_o(output)
        return output

    def __call__(self, queries, keys, values, valid_seq_lens):
        return self.forward(queries, keys, values, valid_seq_lens)

    def transpose_qkv(self, X):
        """
        INPUT: # (batch_size, # queries, # query-size)
        1. # (batch_size, # queries, # num_heads # query-size/num_heads)
        2. # (batch_size, # num_heads, # queries, # query-size/num_heads)
        3. # (batch_size * num_heads, # queries, # query-size/num_heads)
        """
        shape = X.shape
        X = X.reshape(shape[0], shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(-1, X.shape[2], X.shape[3])
        return X

    def transponse_output(self, X):
        """

        Input: X: (batch_size*num_heads, # queries, values/num_heads)
        1.  # (batch_size, # num_heads, # queries, values/num_heads)
        2.  # (batch_size, # queries, # num_heads, values/num_heads)
        3.  # (batch_size, # queries, # values)
        :return:
        """
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        return X

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, querys, keys, values, valid_seq_len):
        ## condition is that the querys and keys are of the same size d
        d = querys.shape[-1]
        features = torch.bmm(querys, keys.transpose(1, 2))/math.sqrt(d)
        attention_weights = masked_softmax(features, valid_seq_len)
        out = torch.bmm(self.dropout(attention_weights), values)
        return out