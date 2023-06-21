import numpy as np
import matplotlib.pyplot as plt



from GRU_seq_models import Seq2Seq, SeqEncoder, SeqDecoder
from trainer import Trainer
from utils import MTFraEng



if __name__ == "__main__":
    batch_size = 8
    lr = 0.005
    gradient_clip_val = 1
    max_epochs = 50
    input_size = 256
    num_hiddens = 256
    num_layers = 2
    dropout = 0.01

    data = MTFraEng(batch_size=batch_size)
    trainer = Trainer(max_epochs=max_epochs, gradient_clip_val=gradient_clip_val)
    source_vocab_size = len(data.src_vocab.token_to_idx)
    target_vocab_size = len(data.tgt_vocab.token_to_idx)
    seq_encoder = SeqEncoder(vocab_size=source_vocab_size, input_size=input_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
    seq_decoder = SeqDecoder(vocab_size=target_vocab_size, input_size=input_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)

    plt.close("all")

    model = Seq2Seq(encoder=seq_encoder, decoder=seq_decoder, pad_token=data.tgt_vocab['<pad>'], lr=lr)
    trainer.fit(data=data, model=model)

    s = np.array([np.mean(trainer.training_loss[idx]) for idx in range(max_epochs)])
    plt.plot(s, label="training_loss")
    s = np.array([np.mean(trainer.validation_loss[idx]) for idx in range(max_epochs)])
    plt.plot(s, label="validation loss")
    plt.legend()
    plt.show()

    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    preds, _ = model.predict_step(data.build(engs, fras), device="cuda:0", num_steps=data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            translation.append(token)
        print(f'{en} => {translation}, bleu,'
              f'{model.bleu(" ".join(translation), fr, k=2):.3f}')