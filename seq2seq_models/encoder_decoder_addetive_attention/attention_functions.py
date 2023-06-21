import torch

def masked_softmax(features, valid_seq_lens):
    # features is 3D matrix
    # valid_seq_len is 1D or 2D matrix
    def _mask_seq(X, valid_seq_len, value=0):
        x_len = X.size(1)
        mask = torch.arange((x_len), device=X.device, dtype=torch.int32)[None, :] < valid_seq_len[:, None]
        X[~mask] = value
        return  X

    if valid_seq_lens is None:
        return torch.nn.functional.softmax(features, dim=-1)
    else:
        shape = features.shape
        if valid_seq_lens.dim() == 1:
            valid_seq_lens = torch.repeat_interleave(valid_seq_lens, shape[1])
        else:
            valid_seq_lens = valid_seq_lens.reshape(-1)
        rescaled_features = _mask_seq(features.reshape(-1, shape[-1]), valid_seq_len=valid_seq_lens, value=-1e6)
        return torch.nn.functional.softmax(rescaled_features.reshape(shape), dim=-1)


class AddetiveAttention(torch.nn.Module):
    def __init__(self, num_hiddens, dropout, bias=False):
        super().__init__()
        self.W_q = torch.nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = torch.nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = torch.nn.LazyLinear(1, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.W_o = torch.nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_seq_len):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # query: (batch_size, num_queries, 1, quereis_size)         # key: (batch_size, 1,  num_key_value_pairs, key_size)
        features = torch.tanh(features) # query: (batch_size, num_queries, 1, quereis_size)         # key: (batch_size, 1,  num_key_value_pairs, key_size)
        features = self.W_v(features).squeeze(-1)
        attention_weights = masked_softmax(features, valid_seq_len)
        output = torch.bmm(self.dropout(attention_weights), values)
        return self.W_o(output)