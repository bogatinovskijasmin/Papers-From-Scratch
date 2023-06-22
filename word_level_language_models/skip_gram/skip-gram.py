import torch
from utils import load_data_ptb

data, vocab = load_data_ptb(batch_size=256, max_window_size=60, num_noise_words=10)
vocab_size = len(vocab.token_to_idx)
embed_size = 100
device = "cuda:0"
max_epochs = 10
lr = 0.002

def skip_gram(central_word, context_and_negative, v, u):
    w_o = u(context_and_negative).permute(0, 2, 1)
    v_c = v(central_word)
    outputs = torch.bmm(v_c, w_o)
    return outputs.reshape(outputs.shape[0], -1)


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, labels, mask):
        return torch.nn.functional.binary_cross_entropy_with_logits(preds.type(torch.float32), labels, mask, reduction="none").mean(dim=1)

net = torch.nn.Sequential(torch.nn.Embedding(vocab_size, embed_size),
                          torch.nn.Embedding(vocab_size, embed_size))

def init_weights(module):
    if isinstance(module, torch.nn.Embedding):
        torch.nn.init.xavier_normal_(module.weight)
net.apply(init_weights)
net.to(device)

loss = Loss()
training_loss = []
optimizer = torch.optim.Adam(net.parameters(), lr=lr)



for epoch in range(max_epochs):
    print(f"Currently running epoch {epoch+1}/{max_epochs}")
    for idx, batch in enumerate(data):
        batch = [a.to(device) for a in batch]
        optimizer.zero_grad()
        central_word, context_and_negative_mask, mask, labels = batch
        out = skip_gram(central_word, context_and_negative_mask, net[0], net[1])
        l = loss(out, labels.type(torch.float32), mask)/mask.sum(axis=1) * mask.shape[1]
        l.sum().backward()
        optimizer.step()
        training_loss.append(l.sum().item())