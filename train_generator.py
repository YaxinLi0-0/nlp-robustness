# ## Preparing Data

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
import sys
import pickle
import pandas as pd

from generator_LSTM import Encoder, Decoder, Seq2Seq
from SeqGAN import train_discriminator, train_gen_epoch, evaluate, inference, generate, train_generator_PG 
from asrdata_utils import get_emb_matrix, idx2seq, batchify, process_feature  
from discriminator import Discriminator
# We'll set the random seeds for deterministic results.

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Next, we download and load the train, validation and test data.
import json
import jieba

true_text = []
DATA = ['0', '1', '2','3','4','5','6','7','8','9']
for d in DATA:
    f = open('data/data_question_'+ d +'.json', 'r')
    data = json.load(f)
    N = len(data['true_message']['data'])

    for i in range(N):
       d = data['true_message']['data'][i]['text']
       true_text.append(d)

#true_text = np.array(true_text)

for i in range(len(true_text)):
    true_str = jieba.lcut(true_text[i], cut_all = False)
    #print(i, true_str)
    if (true_str != []):
        true_text[i] = true_str[0]
        for j in range(1, len(true_str)):
            true_text[i] = true_text[i] + ' ' + true_str[j]

params = {"max_len": 50, 
          "embed_size": 200,
          "num_labels":2,
          "which_emb": 'tencent', 
          "seg_column":"seg_text",
          "epochs": 100, 
          "which":"002",
          "batch_size": 32}

X_true_train = true_text[0:31450]

X_true_dev = true_text[31450: 35381]
X_true_test = true_text[35381:]

#target data

def process_df(path):
    df = pd.read_csv(path)
    df['label'] = df['have_question']
    return df


root_dir = './data/two_class/{}'
df_train = process_df(root_dir.format('train_raw.csv'))
df_dev = process_df(root_dir.format('dev_raw.csv'))
df_test = process_df(root_dir.format('test_raw.csv'))
 
def get_data(df_train, df_dev, df_test, params={}):
    X_train, y_train = 'sos ' + df_train['seg_text'] + ' eos', df_train['label']
    X_dev, y_dev = 'sos ' + df_dev['seg_text'] + ' eos', df_dev['label']
    X_test, y_test = 'sos ' + df_test['seg_text'] + ' eos', df_test['label']
    text_list = X_train.tolist() + X_dev.tolist()

    return X_train, y_train, X_dev, y_dev, X_test, y_test, text_list

X_target_train, y_train, X_target_dev, y_dev, X_target_test, y_test, text_list= get_data(df_train, df_dev, df_test, params = params)

text_list = true_text + text_list

tokenizer, x_target_train, x_true_train, y_train, x_target_dev, x_true_dev, y_dev, x_target_test, x_true_test, y_test, embedding_matrix = process_feature(X_target_train, X_true_train, y_train, X_target_dev, X_true_dev, y_dev, X_target_test, X_true_test, y_test, text_list, params = params)

vocab = tokenizer.word_index
vocab = {v: k for k, v in vocab.items()}
#x_true_train, y_train, x_true_dev, y_dev, x_true_test, y_test, embedding_matrix_true = process_feature(X_true_train, y_train, X_true_dev, y_dev, X_true_test, y_test, true_text, params = params)

PAD_IDX = 0 
BATCH_SIZE = 10
train_data = [(x_true_train[i], x_target_train[i]) for i in range(len(x_true_train))]
valid_data = [(x_true_dev[i], x_target_dev[i]) for i in range(len(x_true_dev))]
test_data = [(x_true_test[i], x_target_test[i]) for i in range(len(x_true_test))]


#print("Preparing batch data.")
#list: [seq_lens, batch size], [batch size]
train_src_batch, train_src_lens, train_tar_batch, train_max_lens = batchify(train_data, BATCH_SIZE, PAD_IDX)
valid_src_batch, valid_src_lens, valid_tar_batch, valid_max_lens = batchify(valid_data, BATCH_SIZE, PAD_IDX)
test_src_batch, test_src_lens, test_tar_batch, test_max_lens = batchify(test_data, BATCH_SIZE, PAD_IDX)

# # Training the Seq2Seq Model
# We then define the encoder, decoder and then our Seq2Seq model, which we place on the `device`.

INPUT_DIM = embedding_matrix.shape[0]
OUTPUT_DIM = embedding_matrix.shape[0]
ENC_EMB_DIM = 200
DEC_EMB_DIM = 200
EMB_DIM = 200
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
#embedding_matrix_true = torch.from_numpy(embedding_matrix_true)

device = torch.device('cuda')

embedding_matrix = torch.from_numpy(embedding_matrix).float().to(device)
enc = Encoder(embedding_matrix, INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(embedding_matrix, OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

gen = Seq2Seq(enc, dec, device).to(device)

dis = Discriminator(embedding_matrix, EMB_DIM, HID_DIM).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


gen.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(gen):,} trainable parameters')

optimizer = optim.Adam(gen.parameters(), lr= 1e-3)

TRG_PAD_IDX = 0

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def word_similarity_loss(output, tar):
     
    return loss


# Next, we'll create a function that we'll use to tell us how long an epoch takes.

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Initialize generator model.

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

"""
for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss = train_gen_epoch(epoch, gen, train_src_batch, train_src_lens, train_tar_batch, train_max_lens, BATCH_SIZE, optimizer, criterion, CLIP)

    valid_loss = evaluate(model, valid_src_batch, valid_src_lens, valid_tar_batch, valid_max_lens, BATCH_SIZE, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './model/pretrain_generator/tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    #print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
"""


gen.load_state_dict(torch.load('./model/pretrain_generator/tut1-model.pt'))

#test_loss, train_gen_batch = inference(gen, train_src_batch, train_src_lens, train_tar_batch, train_max_lens, BATCH_SIZE, criterion)

#valid_loss, valid_gen_batch = inference(gen, valid_src_batch, valid_src_lens, valid_tar_batch, valid_max_lens, BATCH_SIZE, criterion)
import ipdb
ipdb.set_trace()

test_loss, test_gen_batch = inference(gen, test_src_batch, test_src_lens, test_tar_batch, test_max_lens, BATCH_SIZE, criterion)

print(test_loss)
sentence = []
for i in range(len(test_gen_batch)):
    for j in range(10):
        sentence = []
        for k in range(test_max_lens[i]):
            sentence.append(vocab[test_gen_batch[i].view(-1,10).transpose(0,1)[j][k].cpu().detach().item()])
            if (test_gen_batch[i].view(-1,10).transpose(0,1)[j][k].cpu().detach().item() == 4):
               break
        print(sentence)
    for j in range(10):
        sentence = []
        for k in range(test_src_lens[i][j]):
            sentence.append(vocab[test_src_batch[i][j][k].cpu().detach().numpy().item()])
            if (test_src_batch[i][j][k].cpu().detach().numpy().item() == 4):
               break
        print(sentence)	
 

"""
gen.load_state_dict(torch.load('./model/pretrain_generator/generator_epoch3.pt'))

test_loss, train_gen_batch = inference(gen, train_src_batch, train_src_lens, train_tar_batch, train_max_lens, BATCH_SIZE, criterion)

train_gen_batch:[batch, max_len * batch_size]

print(f'| Test Loss: {test_loss:.3f} |')
"""

# Train discriminator

# PRETRAIN DISCRIMINATOR
# random choose 1000 batch to train discriminator

print('\nStarting Discriminator Training...')
dis_optimizer = optim.Adagrad(dis.parameters())

optimizer = optim.Adam(gen.parameters(), lr= 1e-4)

#optimizer = optim.Adagrad(dis.parameters())

#train_discriminator(dis, dis_optimizer, train_gen_batch, train_tar_batch, BATCH_SIZE, 3)
#torch.save(dis.state_dict(), './model/pretrain_discriminator/model.pt')

pretrained_dis_path = './model/pretrain_discriminator/model.pt'
dis.load_state_dict(torch.load(pretrained_dis_path))

# ADVERSARIAL TRAINING
print('\nStarting Adversarial Training...')

import ipdb
ipdb.set_trace()

ADV_TRAIN_EPOCHS = 4
for epoch in range(ADV_TRAIN_EPOCHS):
    print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    # TRAIN GENERATOR
    print('\nAdversarial Training Generator : ', end='')
    sys.stdout.flush()
    gen.train() 
    train_generator_PG(gen, optimizer, dis, train_src_batch, train_src_lens, train_tar_batch, train_max_lens, BATCH_SIZE)
    torch.save(gen.state_dict(), './model/pretrain_generator/generator_epoch'+ str(epoch) +'.pt')

    # GET GENERATED SENTENCES
    test_loss, train_gen_batch = inference(gen, train_src_batch, train_src_lens, train_tar_batch, train_max_lens, BATCH_SIZE, criterion)
    #test_loss, test_gen_batch = inference(gen, test_src_batch, test_src_lens, test_tar_batch, test_max_lens, BATCH_SIZE, criterion)
 
    print('current loss:', test_loss)   
    # TRAIN DISCRIMINATOR
    #print('\nAdversarial Training Discriminator : ')
    #sys.stdout.flush()
    #dis.train()
    #train_discriminator(dis, dis_optimizer, train_gen_batch, train_tar_batch, BATCH_SIZE, 3, 3)
    #torch.save(dis.state_dict(), './model/pretrain_discriminator/discriminator_epoch' + str(epoch) + '.pt')

