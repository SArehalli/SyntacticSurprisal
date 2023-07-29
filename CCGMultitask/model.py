import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, batch_first=False, 
                dropout=0):

        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)


    def init_hidden(self, bsz):
        return (next(self.lstm.parameters()).new_zeros(self.num_layers, bsz, self.hidden_size), 
                next(self.lstm.parameters()).new_zeros(self.num_layers, bsz, self.hidden_size))

    def forward(self, input, hidden):
        embeds = self.embed_dropout(self.embed(input))
        return self.lstm(embeds, hidden)


class Decoder(nn.Module):
    def __init__(self, hidden_size, label_size):
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.linear = nn.Linear(hidden_size, label_size)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)

    def forward(self, input):
        out = self.linear(input)
        out = out.view(-1, self.label_size)
        return F.log_softmax(out, dim=1)

class MultiTaskModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, label_sizes, num_layers, dropout=0):

        super(MultiTaskModel, self).__init__()
        self.lstm = LSTMModel(vocab_size, embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoders = nn.ModuleList([Decoder(hidden_size, label_size) for label_size in label_sizes])
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, bsz):
        return self.lstm.init_hidden(bsz)

    def forward(self, input, hidden):
        lstm_out, hidden = self.lstm(input, hidden)
        lstm_out = self.dropout(lstm_out)
        return [decoder(lstm_out) for decoder in self.decoders] + [hidden]
            


