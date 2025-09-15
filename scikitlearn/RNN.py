import torch
import torch.nn as nn

# Tiny dataset
text = "you are very nice you are very kind you are awesome".split()
vocab = list(set(text))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

# Hyperparams
embed_size, hidden_size = 10, 20
seq_len = 3

# Model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        return self.fc(out), h

model = RNNLM(len(vocab), embed_size, hidden_size)
print(model)
