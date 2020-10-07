
# ## Building the Seq2Seq Model
#
# We'll be building our model in three parts. The encoder, the decoder and a seq2seq model that encapsulates the encoder and decoder and will provide a way to interface with each.
#
# ### Encoder
# In[13]:
import torch
import torch.nn as nn

import spacy
import numpy as np

import random
import math
import time

import pickle
import pandas as pd


class Encoder(nn.Module):
    def __init__(self, embedding_matrix, input_dim, emb_dim, hid_dim, n_layers, dropout, device = torch.device('cuda')):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx = 0)
 
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src_data, src_lens):

        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src_data)).float().to(self.device)

        padded = nn.utils.rnn.pack_padded_sequence(embedded, src_lens)
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(padded)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell


# ### Decoder
#
# Next, we'll build our decoder, which will also be a 2-layer (4 in the paper) LSTM.
#
# ![](assets/seq2seq3.png)
#

class Decoder(nn.Module):
    def __init__(self, embedding_matrix, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        #self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx = 0)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        input = input.unsqueeze(0)
         
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, src_lens, trg, trg_lens, teacher_forcing_ratio = 1.0):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg_lens
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, src_lens)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output     #[bsz, output_dim]
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs

    def batchPGLoss(self, inp, seq_len, target, max_len, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
        Inputs: inp, target
            - inp: seq_len x batch_size
            - target: seq_len x batch_size
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)
            inp should be target with <s> (start letter) prepended
        """

        batch_size = inp.size()[1]
        #inp = inp.permute(1, 0)          # seq_len x batch_size
        #target = target.permute(1, 0)    # seq_len x batch_size
        #h = self.init_hidden(batch_size)
	
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(inp, seq_len)
        input = target[0]
             
        loss = 0
        for i in range(1, max_len):
            #out = self.forward(inp[i], h)
            #receive output tensor (predictions) and new hidden and cell states 
            out, hidden, cell = self.decoder(input, hidden, cell)
            input = out.argmax(1)
            
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size

