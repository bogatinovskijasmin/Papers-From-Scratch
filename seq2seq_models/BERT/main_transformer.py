import collections

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.modules.loss

from attention_model import BERT
from BERT_pretraining_data_preparation import load_data_wiki



if __name__ == "__main__":
    batch_size = 512
    lr = 0.01

    max_epochs = 50
    num_heads = 4
    num_hiddens = 128
    num_layers = 2
    dropout = 0.2
    ffn_input = 256
    use_bias = True
    device = "cuda:0"

    data, vocab = load_data_wiki(batch_size=batch_size, max_len=64)

    BERT = BERT(num_layers=num_layers, num_hiddens=num_hiddens, dropout=dropout,num_heads=num_heads, bias=use_bias, ffn_input=ffn_input, vocab_size=len(vocab.token_to_idx))
    BERT.to(device)

    loss = torch.nn.modules.loss.CrossEntropyLoss()
    optimizer = torch.optim.Adam(BERT.parameters(), lr=lr)

    def get_loss(mlm_output, nsp_output, all_mlm_labels, labels_nsp, masked_positions, loss):
        l_nsp = loss(nsp_output, labels_nsp)
        l_mlm = loss(mlm_output.reshape(-1, mlm_output.shape[-1]), all_mlm_labels.reshape(-1))
        l = l_mlm + l_nsp
        return l, l_mlm, l_nsp

    plt.close("all")
    training_loss_nsp = collections.defaultdict(list)
    training_loss_mlm = collections.defaultdict(list)

    for epoch in range(max_epochs):
        print(f"Currently running epoch {epoch+1}/{max_epochs}")

        for idx, batch in enumerate(data):
            batch = [a.to(device) for a in batch]

            encoded_sequence, segment, valid_seq_len, all_pred_positions, masked_positions, all_mlm_labels, labels_nsp = batch
            optimizer.zero_grad()

            X_encoded, mlm_output, nsp_output = BERT.forward(X=encoded_sequence, segment=segment, valdid_seq_len=valid_seq_len, pred_pos=all_pred_positions)
            l_, l_mlm_, l_nsp_ = get_loss(mlm_output, nsp_output, all_mlm_labels, labels_nsp, masked_positions, loss)
            l_.backward()
            optimizer.step()

            training_loss_mlm[epoch].append(l_mlm_.item())
            training_loss_nsp[epoch].append(l_nsp_.item())


    s = [np.mean(training_loss_mlm[idx]) for idx in range(max_epochs)]
    plt.plot(s, label="mlm loss")
    s = [np.mean(training_loss_nsp[idx]) for idx in range(max_epochs)]
    plt.plot(s, label="nsp loss")