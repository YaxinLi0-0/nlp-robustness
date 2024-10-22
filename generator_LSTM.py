
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
    def __init__(self, encoder, decoder, device, use_pinyin = False):
        super().__init__()

        self.use_pinyin = use_pinyin 
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, src_lens, trg, trg_lens, vocab_sim, teacher_forcing_ratio = 1.0):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_lens, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, src_lens)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        softmax = nn.Softmax(dim = 1)
        for t in range(1, trg_lens):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            if (self.use_pinyin == 'True'):           
              for i in range(batch_size):
                 #print('output' + str(i))
                 for j in range(trg_vocab_size - 1):
                     min_dis = 20
                     if (j == 3 or j==4):
                        min_dis = 0
                     else:
                        for k in range(max(0, t-1), min(t + 1, trg_lens)):
                            if ( vocab_sim[src[k][i]][j] < min_dis):
                               min_dis = vocab_sim[src[k][i]][j]
                    
                     output[i][j] = output[i][j] * (min_dis * min_dis - 20 * min_dis + 1)
            
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

    def batchPGLoss(self, inp, seq_len, trg, trg_len, num, dis):
        """
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: seq_len x batch_size
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)
            inp should be target with <s> (start letter) prepended
        """

        batch_size = inp.size()[1]
        #h = self.init_hidden(batch_size)
        #import ipdb
        #ipdb.set_trace()
        preds = self.forward(inp, seq_len, trg, trg_len)
        output = preds.argmax(-1)
        rewards = self.get_reward(output, seq_len, trg_len, batch_size, num, dis)
        loss = GANLoss(output, rewards)
        return loss/batch_size
    
    def GANLoss(self, output, rewards):
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        one_hot = one_hot.to(self.device)
        
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss

    def sample(self, inp, seq_len, max_len, batch_size, x=None):
        softmax = nn.Softmax()
        res = []
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = torch.zeros((batch_size, 1)).long()
        x = x.to(self.device)
        hidden, cell = self.encoder(inp, seq_len)
        samples = []
        if flag:
            for i in range(max_len):
                output, hidden, cell = self.decoder(x, hidden, cell)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(0)
            lis = x.chunk(x.size(0), dim=1)
            for i in range(given_len):
                output, hidden, cell = self.decoder(lis[i][0], hidden, cell)
                samples.append(lis[i])
            #import ipdb
            #ipdb.set_trace()
            output = softmax(output)
            x = output.multinomial(1)
            for i in range(given_len, max_len):
                samples.append(x)
                output, hidden, cell = self.decoder(x, hidden, cell)
                output = softmax(output)
                x = output.multinomial(1)
        
        output = torch.cat(samples, dim=1)
        return output
    
    def get_reward(self, x, seq_len, max_len, batch_size, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        Returns:
            rewards: mean rewards for each token, [seq_len * bsz * 1]
        """
        rewards = []
        #batch_size = x.size(0)
        #seq_len = x.size(1)
        for i in range(num):
            for l in range(1, max_len):
                state_data = x[0:l, :]
                samples = self.sample(x, seq_len, max_len, batch_size, state_data)
                h = discriminator.init_hidden(samples.shape[1])    
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards


