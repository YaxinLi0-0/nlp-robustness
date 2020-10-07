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
import jieba

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# In[5]:


def init_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[6]:


def process_df(path):
    df = pd.read_csv(path)
    df['label'] = df['have_question']
    return df

def load_data():
    root_dir = './data/two_class/{}'
    df_train = process_df(root_dir.format('train_raw.csv'))
    df_dev = process_df(root_dir.format('dev_raw.csv'))
    df_test = process_df(root_dir.format('test_raw.csv'))
    return df_train,df_dev,df_test


# ## 数据准备

# In[7]:


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

def process_feature(X_train, y_train, X_dev, y_dev, X_test, y_test, X_true_test, text_list, params={}):
    seg_column = params['seg_column']
    max_len = params['max_len']

    embedding_matrix, tokenizer = get_emb_matrix(text_list, params=params)
    # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_dev_word_ids = tokenizer.texts_to_sequences(X_dev)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    X_true_test_word_ids = tokenizer.texts_to_sequences(X_true_test)
    
    # 序列模式
    x_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=max_len)
    x_dev_padded_seqs = pad_sequences(X_dev_word_ids, maxlen=max_len)
    x_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=max_len)
    x_true_test_padded_seqs = pad_sequences(X_true_test_word_ids, maxlen=max_len)
    
    # get label
    num_labels = params['num_labels']
    y_train_2 = to_categorical(y_train, num_labels)
    y_dev_2 = to_categorical(y_dev, num_labels)
    y_test_2 = to_categorical(y_test, num_labels)

    return x_train_padded_seqs, y_train_2, x_dev_padded_seqs, y_dev_2, x_test_padded_seqs, y_test_2, x_true_test_padded_seqs, embedding_matrix

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


def save_model(model,params, model_name):
    which = params['which']
    which_emb = params['which_emb']
    model_path = 'model/二分类/bi-lstm/baseline/bi-lstm_{}_{}_{}.model'.format(which,which_emb, model_name)
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
df_train,df_dev,df_test = load_data()

"""
Read true text
"""
import json

true_text = []
DATA = ['0', '1', '2','3','4','5','6','7','8','9']
for d in DATA:
    f = open('data/data_question_'+ d +'.json', 'r')
    data = json.load(f)
    N = len(data['true_message']['data'])

    for i in range(N):
       d = data['true_message']['data'][i]['text'] 
       true_text.append(d)

# In[15]:

true_text = np.array(true_text)

for i in range(true_text.size):
    true_str = jieba.lcut(true_text[i], cut_all = False)
    #print(i, true_str)
    if (true_str != []):
        true_text[i] = true_str[0]
        for j in range(1, len(true_str)):
            true_text[i] = true_text[i] + ' ' + true_str[j]

params = {"max_len": 60, 
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


# In[16]:
X_train, y_train, X_dev, y_dev, X_test, y_test, text_list= get_data(df_train, df_dev, df_test, params = params)

same_pinyin = get_same_pinyin_vocabulary('./same_pinyin.txt')
word_freq = get_word_freq('./chinese-words.txt')

"""
Add perturbation
"""
#ipdb.set_trace()
#for i in range(X_train.size):
#    X_train[i] = replace_samePinyin(X_train[i], same_pinyin, word_freq)
#    X_train[i] = X_train[i][1:-1]
    #print(X_test[i])

x_train, y_train, x_dev, y_dev, x_test, y_test,x_true_test, embedding_matrix = process_feature(X_train, y_train, X_dev, y_dev, X_test, y_test, X_true_test, text_list, params = params)

#x_true_train, y_train, x_true_dev, y_dev, x_true_test, y_test, x_test, embedding_matrix = process_feature(X_true_train, y_train, X_true_dev, y_dev, X_true_test, y_test, X_train, true_text, params = params)

# In[74]:
#tokenizer = params['tokenizer']
#token_path = 'model/二分类/bi-lstm/baseline/tokenizer.plk'
#with open(token_path,'wb') as f:
#    pickle.dump(tokenizer,f)


# In[75]:


model = train_model(x_train, y_train, x_dev, y_dev, embedding_matrix, params)


# In[53]:
# In[76]:

#model = get_model(embedding_matrix, params=params)
#model = load_model("model/二分类/bi-lstm/checkpoint/best_clean.hdf5")
#import ipdb
#ipdb.set_trace()
y_true_pred = model.predict(x_true_test)[:, 1]
y_pred = model.predict(x_test)[:,1]          

print_metris(getModelInfo(y_test.argmax(axis=1), y_true_pred))                                          
print_metris(getModelInfo(y_test.argmax(axis=1), y_pred))

name = 'asr_train'
save_model(model,params, name)


"""visualization"""
"""
#lda = LinearDiscriminantAnalysis(n_components=2)
#lda.fit(x_train, y_train)
#pca = PCA(n_components=2)
#pca.fit(x_train)

tsne = TSNE(n_components = 2, learning_rate = 500, perplexity = 100)
decomposition_data = tsne.fit_transform(x_train)

x = []
y = []

y_train_label = np.argmax(y_train,axis = 1) 
for i in decomposition_data:
    x.append(i[0])
    y.append(i[1])

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
plt.scatter(x, y, c=y_train_label, marker="x")
plt.xticks(())
plt.yticks(())
# plt.show()
plt.savefig('./clean.png', aspect=1)

# In[ ]:
"""



