import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):

    def __init__(self, embedding_matrix, emb_dim, hidden_dim, device='cuda', vocab_size = 0, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        #self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        #self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, padding_idx = 0)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        #Init hidden layer
        h = torch.zeros(2*2*1, batch_size, self.hidden_dim).to(self.device)
        return h

    def forward(self, input, hidden):
        # input dim                                                # [ seq_len x batch_size ]
        emb = self.embeddings(input)                               # [ seq_len x batch_size x embedding_dim]
        #emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp, bsz):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: seq_len x batch_size

        Returns: out
            - out: batch_size ([0,1] score)
        """
   
        h = self.init_hidden(bsz)
        inp = inp.view(-1, bsz)
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)

