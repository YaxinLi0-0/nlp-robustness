from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import discriminator
import helpers


CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

def train_gen_epoch(epoch, model, src_data, src_lens, tar_data, max_lens, vocab_sim, bsz, optimizer, criterion, clip, device = 'cuda'):

    model.train()

    epoch_loss = 0

    for i in range(len(src_data)):
        src = src_data[i].transpose(0,1).to(device)
        tar = tar_data[i].transpose(0,1).to(device)
        #print(src.shape, tar.shape)
        optimizer.zero_grad()
        #print(i)
        
        output = model(src, src_lens[i], tar, max_lens[i], vocab_sim)

        #tar = [tar len, batch size]
        #output = [tar len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tar = tar[1:].reshape(-1)

        #tar = [(tar len - 1) * batch size]
        #output = [(tar len - 1) * batch size, output dim]
        #print(output.shape, tar.shape) 
        loss = 1 * criterion(output, tar)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        #print(output.argmax(-1), tar)
        """
        if ((epoch + 1) % 2 == 0 and (i % 100) == 0):
           out = np.array(idx2seq(output.argmax(1), vocab))
           out_tar = np.array(idx2seq(tar, vocab))
           #out_src = np.array(idx2seq(src, vocab))
           for j in range(bsz):
               word = [k*bsz+j for k in range(max_lens[i] - 1)]
               word = np.array(word)
               print(out[word], out_tar[word])
        """
    return epoch_loss / bsz

def evaluate(model, src_data, src_lens, tar_data, max_lens, vocab_sim, bsz, criterion, device = 'cuda'):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i in range(len(src_data)):
            src = src_data[i].transpose(0, 1).to(device)
            tar = tar_data[i].transpose(0, 1).to(device)

            output = model(src, src_lens[i], tar, max_lens[i], vocab_sim) #turn off teacher forcing
            #tar = [tar len, batch size]
            #output = [tar len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tar = tar[1:].reshape(-1)
            #tar = [(tar len - 1) * batch size]
            #output = [(tar len - 1) * batch size, output dim]

            loss = criterion(output, tar)

            epoch_loss += loss.item()

    return epoch_loss / bsz

def inference(model, src_data, src_lens, tar_data, max_lens, vocab_sim, bsz, criterion, device = 'cuda'):

    model.eval()

    epoch_loss = 0
    
    asr_gen = []
    with torch.no_grad():

        for i in range(len(src_data)):
            src = src_data[i].transpose(0, 1).to(device)    #src: [seq_len, batch_size]
            tar = tar_data[i].transpose(0, 1).to(device)    #tar: [seq_len, batch_size]
            
            output = model(src, src_lens[i], tar, max_lens[i], vocab_sim, teacher_forcing_ratio=0) #turn off teacher forcing

            #tar = [tar len, batch size]
            #output = [tar len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)   #rm <sos>, calculate loss
            tar = tar[1:].reshape(-1)

            asr_gen.append(output.argmax(-1))             #[seq_len x batch_size]
            #tar = [(tar len - 1) * batch size]
            #output = [(tar len - 1) * batch size, output dim]

            loss = criterion(output, tar)

            epoch_loss += loss.item()
            
    return epoch_loss / bsz, asr_gen

def generate(model, src, src_len, tar, max_len, bsz, criterion):
 
    model.eval()
    epoch_loss = 0
    device = model.device

    with torch.no_grad():

       output = model(src, src_len, tar, max_len, teacher_forcing_ratio=0) #turn off teacher forcing

       output_dim = output.shape[-1]
       output = output[1:].view(-1, output_dim)   #rm <sos>, calculate loss
       tar = tar[1:].reshape(-1)

       #asr_gen.append(output.argmax(-1))             #[seq_len x batch_size]

       loss = criterion(output, tar)

       epoch_loss += loss.item()

    return epoch_loss / bsz, output.argmax(-1)

def train_generator_PG(gen, gen_opt, dis, train_src_batch, train_src_lens, train_tar_batch, train_max_lens, bsz):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    total_loss = 0
    num_batches = len(train_src_batch)
    device = gen.device
    
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    
    for i in range(num_batches):
        
        src_lens = train_src_lens[i]
        max_lens = train_max_lens[i]

        inp, tar = train_src_batch[i].transpose(0, 1).to(device), train_tar_batch[i].transpose(0,1).to(device)   # inp: [seq_lens x bsz ] 

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, src_lens, tar, max_lens, 2, dis)
        #print('current batch:', i, 'pg_loss:', pg_loss)
        pg_loss.backward()
        gen_opt.step()
        total_loss = total_loss + pg_loss
    print('Finish training generator. Total loss:', total_loss/(num_batches * bsz))
 
def train_discriminator(discriminator, dis_opt, train_tar_batch, train_asr_batch, bsz = 10, d_steps = 5, epochs = 5):

    # generating a small validation set before training (using oracle and generator)
    device = discriminator.device
    POS_NEG_SAMPLES = len(train_tar_batch)
    BATCH_SIZE = bsz
    for d_step in range(d_steps):
        #s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        #dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        random_batch = [np.random.randint(POS_NEG_SAMPLES) for i in range(1000)] 
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in random_batch:
                inp_gen, inp_asr = train_tar_batch[i].view(-1, bsz).to(device), train_asr_batch[i].transpose(0,1)[1:].to(device)  
                inp = torch.cat((inp_gen, inp_asr),1)
                target = np.concatenate((np.zeros(bsz), np.ones(bsz)))
                target = torch.from_numpy(target).float().to(device)

                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp, 2 * bsz)
                
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % np.ceil(np.ceil(POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= np.ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= 1000 * 2 * bsz

            #val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((out>0.5)==(target>0.5)).data.item()/(2*bsz)))


