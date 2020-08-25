#!/usr/bin/env python
# coding: utf-8

# # 基本处理

# In[1]:


import os
import sys
sys.path.append('/tabchen_utils/tabchen_utils.py')
from tabchen_utils import getModelInfo
import ipdb
import pickle
# In[2]:


from replace_word import *
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model

# In[3]:


def print_metris(report):
    columns = ['Accuracy','Precision','Recall','F_meansure','AUC_Value']
    print('\t'.join([str(report[x]) for x in columns]))


# # 深度网络

# In[4]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional,GlobalMaxPool1D
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import *
import numpy as np
import gensim
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
def show_model(model):
    return SVG(model_to_dot(model,show_shapes=True,show_layer_names=False).create(prog='dot', format='svg'))


# In[5]:


def init_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[6]:

def load_data():
    root_dir = './{}'
    f = open('asr_text_417.txt','r')
    
    asr_data = f.readlines()
    f = open('true_text_417.txt', 'r')
    true_data = f.readlines()
    return asr_data, true_data


# ## 数据准备

# In[7]:


def get_emb_matrix(text_list,params={}):
    import ipdb
    ipdb.set_trace()
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
        if word in w2v_emb:
            embedding_vector = w2v_emb[word]
            embedding_matrix[i, :] = embedding_vector
            in_num += 1
    return embedding_matrix,tokenizer


# In[8]:

def get_data(df_train, df_dev, df_test, params={}):
    X_train, y_train = df_train['seg_text'], df_train['label']
    X_dev, y_dev = df_dev['seg_text'], df_dev['label']
    X_test, y_test = df_test['seg_text'], df_test['label']
    text_list = df_train['seg_text'].tolist() + df_dev['seg_text'].tolist()

    return X_train, y_train, X_dev, y_dev, X_test, y_test, text_list

def process_feature(asr, true, params={}):
    text_list = asr+ true 
    seg_column = params['seg_column']
    max_len = params['max_len']

    embedding_matrix, tokenizer = get_emb_matrix(text_list, params=params)
    # 将每个词用词典中的数值代替
    asr_ids = tokenizer.texts_to_sequences(asr)
    true_ids = tokenizer.texts_to_sequences(true)
    #X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    # 序列模式
    asr_padded_seqs = pad_sequences(asr_ids, maxlen=max_len)
    true_padded_seqs = pad_sequences(true_ids, maxlen=max_len)
    #x_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=max_len)

    return asr_ids, true_ids, asr_padded_seqs, true_padded_seqs, embedding_matrix
# ## bi-LSTM

# In[9]:


# def get_model(embedding_matrix,params={}):
    
#     max_len = params['max_len']
#     embed_size = params['embed_size']
    
#     input_1 = Input(shape=(max_len,))
#     x = Embedding(len(params['vocab'])+1, embed_size,
#                   weights=[embedding_matrix], trainable=False)(input_1)
#     x = Bidirectional(
#         LSTM(32, dropout=0.5, recurrent_dropout=0.1, return_sequences=True))(x)
#     x = GlobalMaxPool1D()(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.2)(x)
#     x = Dense(32, activation="relu", name='z_2')(x)
#     x = Dense(params['num_labels'], activation='softmax')(x)
#     model = Model(inputs=input_1, outputs=x)
#     model.compile(optimizer='adam', loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

def get_model(embedding_matrix,params={}):
    
    max_len = params['max_len']
    embed_size = params['embed_size']
    
    input_1 = Input(shape=(max_len,))
    x = Embedding(len(params['vocab'])+1, embed_size,
                  weights=[embedding_matrix], trainable=False)(input_1)
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(128,dropout=0.2,return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
    x = Dense(64, activation="relu", name='z_2')(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(params['num_labels'], activation='softmax')(x)
    model = Model(inputs=input_1, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[10]:


def get_callback(params):
    filepath="model/二分类/bi-lstm/checkpoint/best.hdf5"
    init_dir(filepath)
    early_stopping_monitor = EarlyStopping(patience=5,verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',period=1)
    return [early_stopping_monitor,checkpoint],filepath


# In[11]:


def save_model(model,params):
    which = params['which']
    which_emb = params['which_emb']
    model_path = 'model/二分类/bi-lstm/baseline/bi-lstm_{}_{}.model'.format(which,which_emb)
    init_dir(model_path)
    with open(model_path,'wb') as f:
        pickle.dump(model,f)


# In[12]:


def train_model(x_train, y_train,x_dev, y_dev,embedding_matrix, params):
    model = get_model(embedding_matrix, params=params)
    callbacks, best_model_path = get_callback(params)
    epochs = params['epochs']
    batch_size = params['batch_size']
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_dev, y_dev), callbacks=callbacks)
    model.load_weights(best_model_path)
    return model


# # 模型训练

# In[13]:

asr_data, true_data = load_data()
for i in range(len(asr_data)):
    for j in range(17):
        if asr_data[i][j] == ' ':
           asr_data[i] = asr_data[i][j + 1:-2]
           break
    for j in range(17):
        if true_data[i][j] == ' ':
           true_data[i] = true_data[i][j + 1: -2]
           break

# In[14]:


# In[15]:


params = {"max_len": 60, 
          "embed_size": 200,
          "num_labels":2,
          "which_emb": 'tencent', 
          "seg_column":"seg_text",
          "epochs": 100, 
          "which":"002",
          "batch_size": 32}


# In[16]:
#X_train, y_train, X_dev, y_dev, X_test, y_test, text_list= get_data(df_train, df_dev, df_test, params = params)

same_pinyin = get_same_pinyin_vocabulary('./same_pinyin.txt')
word_freq = get_word_freq('./chinese-words.txt')
#ipdb.set_trace()
a_id, t_id, a_seq, t_seq, embedding_matrix = process_feature(asr_data, true_data, params = params)

print('asr:\n', a_seq)
print('true:\n', t_seq)

all_seq = np.concatenate((a_seq, t_seq), axis = 0)

"""cluster and visualization"""
"""
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--perp', type = int, default = 50)
args = parser.parse_args()

tsne = TSNE(n_components = 3, learning_rate = 500, perplexity = args.perp)
decomposition_data = tsne.fit_transform(all_seq)


x = []
y = []
z = []
y_train_label = np.concatenate((np.zeros(417), np.ones(417)), axis = 0) 
for i in decomposition_data:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c=y_train_label, marker="x")
plt.xticks(())
plt.yticks(())
# plt.show()
plt.savefig('./clean_pert.png', aspect=1)
"""

"""chinese tone"""
ipdb.set_trace()
cal_0 = 0
cal_1 = 0 
from ChineseTone import *
for i in range(417):
    l = min(len(asr_data[i]), len(true_data[i]))
    
    asr_pinyin = PinyinHelper.convertToPinyinFromSentence(asr_data[i][0:l], pinyinFormat=PinyinFormat.WITHOUT_TONE)
    true_pinyin = PinyinHelper.convertToPinyinFromSentence(true_data[i][0:l], pinyinFormat=PinyinFormat.WITHOUT_TONE)
    if(asr_data[i] == true_data[i]):
      cal_1 = cal_1 + 1
    if(asr_pinyin == true_pinyin):
      cal_0 = cal_0 + 1

print(cal_0, cal_1) 
