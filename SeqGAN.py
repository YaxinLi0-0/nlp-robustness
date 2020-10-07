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


CUDA = False
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
def train_gen_epoch(epoch, model, src_data, src_lens, tar_data, max_lens, bsz, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i in range(len(src_data)):
        src = src_data[i].transpose(0,1).to(device)
        tar = tar_data[i].transpose(0,1).to(device)
        #print(src.shape, tar.shape)
        optimizer.zero_grad()
        #print(i)
        
        output = model(src, src_lens[i], tar, max_lens[i])

        #tar = [tar len, batch size]
        #output = [tar len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tar = tar[1:].reshape(-1)

        #tar = [(tar len - 1) * batch size]
        #output = [(tar len - 1) * batch size, output dim]
        #print(output.shape, tar.shape) 
        loss = word_similarity(output, tar) * criterion(output, tar)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        #import ipdb
        #ipdb.set_trace()
        #print(output.argmax(-1), tar)
        if ((epoch + 1) % 2 == 0 and (i % 100) == 0):
           out = np.array(idx2seq(output.argmax(1), vocab))
           out_tar = np.array(idx2seq(tar, vocab))
           #out_src = np.array(idx2seq(src, vocab))
           for j in range(bsz):
               word = [k*bsz+j for k in range(max_lens[i] - 1)]
               word = np.array(word)
               print(out[word], out_tar[word])
        
    return epoch_loss / bsz

def evaluate(model, src_data, src_lens, tar_data, max_lens, bsz, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i in range(len(src_data)):
            src = src_data[i].transpose(0, 1).to(device)
            tar = tar_data[i].transpose(0, 1).to(device)

            output = model(src, src_lens[i], tar, max_lens[i]) #turn off teacher forcing
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

def inference(model, src_data, src_lens, tar_data, max_lens, bsz, criterion, device = 'cuda'):

    model.eval()

    epoch_loss = 0
    
    asr_gen = []
    with torch.no_grad():

        for i in range(len(src_data)):
            src = src_data[i].transpose(0, 1).to(device)    #src: [seq_len, batch_size]
            tar = tar_data[i].transpose(0, 1).to(device)    #tar: [seq_len, batch_size]
            
            output = model(src, src_lens[i], tar, max_lens[i], teacher_forcing_ratio=0) #turn off teacher forcing

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

         

# Our evaluation loop is similar to our training loop, however as we aren't updating any parameters we don't need to pass an optimizer or a clip value.
#

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))

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
        
        #s = gen.sample(BATCH_SIZE*2)        # 64 works best
        #inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        
        src_lens = train_src_lens[i]
        max_lens = train_max_lens[i]

        inp, tar = train_src_batch[i].transpose(0, 1).to(device), train_tar_batch[i].transpose(0,1).to(device)   # inp: [seq_lens x bsz ] 
        #import ipdb
        #ipdb.set_trace()   
        rewards = dis.batchClassify(tar, bsz)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, src_lens, tar.view(-1, bsz), max_lens, rewards)
        #print('current batch:', i, 'pg_loss:', pg_loss)
        pg_loss.backward()
        gen_opt.step()
        total_loss = total_loss + pg_loss
    # sample from generator and compute oracle NLL
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                               start_letter=START_LETTER, gpu=CUDA)
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


# MAIN
if __name__ == '__main__':
    oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    oracle.load_state_dict(torch.load(oracle_state_dict_path))
    oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        oracle_samples = oracle_samples.cuda()

    # GENERATOR MLE TRAINING

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 50, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                               start_letter=START_LETTER, gpu=CUDA)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 5, 3)