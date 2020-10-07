from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

import numpy as np
import pickle
import torch

def get_emb_matrix(text_list,params={}):
    tokenizer = Tokenizer(split=" ")
    tokenizer.fit_on_texts(text_list)
    vocab = tokenizer.word_index
    params['vocab'] = vocab
    params['tokenizer'] = tokenizer
    
    w2v_emb = pickle.load(
        open('embeddings/{}_small.plk'.format(params['which_emb']), 'rb'))
    in_num = 0
    embedding_matrix = np.zeros((len(vocab) + 1, 200))
    for word, i in vocab.items():
        if word == 'eos' or word == 'sos':
            embedding_vector = np.random.rand(200) * 2 - 1
            embedding_matrix[i, :] = embedding_vector
            in_num += 1
           
        elif word in w2v_emb:
            embedding_vector = w2v_emb[word]
            embedding_matrix[i, :] = embedding_vector
            in_num += 1
    return embedding_matrix,tokenizer

def idx2seq(seq_list, vocab):
    sentence = []
    for id in seq_list:
        if not (id == 0):
           sentence.append(vocab[id.cpu().numpy().item()])
        else:
           sentence.append(' ')
    return sentence

def process_feature(X_train, X_target_train, y_train, X_dev, X_target_dev, y_dev, X_test, X_target_test, y_test, text_list, params={}):
    seg_column = params['seg_column']
    max_len = params['max_len']

    embedding_matrix, tokenizer = get_emb_matrix(text_list, params=params)
    # 将每个词用词典中的数值代替
    #import ipdb
    #ipdb.set_trace()
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_dev_word_ids = tokenizer.texts_to_sequences(X_dev)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    
    X_train_target_word_ids = tokenizer.texts_to_sequences(X_target_train)
    X_dev_target_word_ids = tokenizer.texts_to_sequences(X_target_dev)
    X_test_target_word_ids = tokenizer.texts_to_sequences(X_target_test)
    
    # get label
    num_labels = params['num_labels']
    y_train = to_categorical(y_train, num_labels)
    y_dev = to_categorical(y_dev, num_labels)
    y_test = to_categorical(y_test, num_labels)

    return tokenizer, X_train_word_ids, X_train_target_word_ids, y_train, X_dev_word_ids, X_dev_target_word_ids, y_dev, X_test_word_ids, X_test_target_word_ids, y_test, embedding_matrix

def batchify(data, bsz, pad_idx):
    # data: list of tuples (xs, ys)
    ntenbatch = len(data) // (bsz*10) 
    src_batchdata = []
    src_lens = []
    tar_batchdata = []
    tar_lens = []
    max_lens = []
    for i in range(ntenbatch):
        #print('preparing batch ', i, ' data')
        tenbatch = data[i * (bsz * 10):(i + 1) * (bsz * 10)]
        sorted_tenbatch = sorted(tenbatch, key=lambda d: len(d[0]), reverse=True) # decreasing order according to the length of the input sequence 
        for j in range(len(sorted_tenbatch) // bsz):
            max_len = len(sorted_tenbatch[j * bsz][0])
            max_len_1 = len(sorted_tenbatch[j * bsz][1])
            for k in range(bsz):
               if len(sorted_tenbatch[j * bsz + k][1]) > max_len_1:
                  max_len_1 = len(sorted_tenbatch[j * bsz + k][1])
            if max_len_1 > max_len:
               max_len = max_len_1
            x_lens = [len(ins[0]) for ins in sorted_tenbatch[j * bsz: (j + 1) * bsz]]
            #y_lens = [len(ins[1]) for ins in sorted_tenbatch[j * bsz: (j + 1) * bsz]]
            y_lens = x_lens
            for k in range(1, bsz):
                if x_lens[k] == 0:
                    sorted_tenbatch[j * bsz + k] = sorted_tenbatch[j * bsz + (k-1)]
                    x_lens[k] = x_lens[k-1]

            xs = [ins[0] + [pad_idx] * (max_len - len(ins[0])) for ins in sorted_tenbatch[j * bsz: (j + 1) * bsz]]
            ys = [ins[1] + [pad_idx] * (max_len - len(ins[1])) for ins in sorted_tenbatch[j * bsz: (j + 1) * bsz]]

            xs_batch = torch.LongTensor(xs)
            x_lens_batch = torch.LongTensor(x_lens)
            ys_batch = torch.LongTensor(ys)
            y_lens_batch = torch.LongTensor(y_lens)

            src_batchdata.append(xs_batch)
            src_lens.append(x_lens_batch)
            tar_batchdata.append(ys_batch)
            #tar_lens.append(y_lens_batch)
            max_lens.append(max_len)
    return src_batchdata, src_lens, tar_batchdata, max_lens




