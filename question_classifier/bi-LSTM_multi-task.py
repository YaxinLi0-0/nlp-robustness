#!/usr/bin/env python
# coding: utf-8

# # 基本处理

# In[3]:


import os
import sys
sys.path.append('/share/tools/tabchen_utils')
from tabchen_utils import *


# In[4]:


from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot


# In[5]:


# def print_metris(report):
#     columns = ['Accuracy','Precision','Recall','F_meansure','AUC_Value']
#     print('\t'.join([str(report[x]) for x in columns]))


# # 深度网络

# In[6]:


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


# In[7]:


def init_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[8]:


que_label_map = {'其它问句': 0, '自问自答问句': 1, '认知问句': 2, '评估问句': 3, '非提问': 4}
def process_df(path):
    df = pd.read_csv(path)
    df['label'] = df['question_label'].map(que_label_map)
    return df

def load_data():
    root_dir = 'data/多分类/{}'
    df_train = process_df(root_dir.format('train_raw.csv'))
    df_dev = process_df(root_dir.format('dev_raw.csv'))
    df_test = process_df(root_dir.format('test_raw.csv'))
    return df_train,df_dev,df_test


# ## 数据准备

# In[9]:


def get_emb_matrix(text_list,params={}):
    tokenizer = Tokenizer(split=" ")
    tokenizer.fit_on_texts(text_list)
    vocab = tokenizer.word_index
    params['vocab'] = vocab
    params['tokenizer'] = tokenizer
    w2v_emb = pickle.load(
        open('data/多分类/{}_small.plk'.format(params['which_emb']), 'rb'))
    in_num = 0
    embedding_matrix = np.zeros((len(vocab) + 1, 200))
    for word, i in vocab.items():
        if word in w2v_emb:
            embedding_vector = w2v_emb[word]
            embedding_matrix[i, :] = embedding_vector
            in_num += 1
    return embedding_matrix,tokenizer


# In[39]:


def convert_multi_label(y):
    return [y[:,i] for i in range(5)]


# In[40]:


def process_feature(df_train, df_dev, df_test, params={}):
    seg_column = params['seg_column']
    max_len = params['max_len']
    X_train, y_train = df_train['seg_text'], df_train['label']
    X_dev, y_dev = df_dev['seg_text'], df_dev['label']
    X_test, y_test = df_test['seg_text'], df_test['label']
    text_list = df_train['seg_text'].tolist() + df_dev['seg_text'].tolist()
    embedding_matrix, tokenizer = get_emb_matrix(text_list, params=params)
   # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_dev_word_ids = tokenizer.texts_to_sequences(X_dev)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    # 序列模式
    x_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=max_len)
    x_dev_padded_seqs = pad_sequences(X_dev_word_ids, maxlen=max_len)
    x_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=max_len)
    # get label
    num_labels = params['num_labels']
    y_train_2 = convert_multi_label(to_categorical(y_train, num_labels))
    y_dev_2 = convert_multi_label(to_categorical(y_dev, num_labels))
    y_test_2 = convert_multi_label(to_categorical(y_test, num_labels))

    return x_train_padded_seqs, y_train_2, x_dev_padded_seqs, y_dev_2, x_test_padded_seqs, y_test_2, embedding_matrix


# ## bi-LSTM

# In[50]:


def get_model(embedding_matrix, params={}):
    max_len = params['max_len']
    embed_size = params['embed_size']
    x = Input(shape=(max_len,))
    shared = Embedding(len(params['vocab'])+1, embed_size,
                       weights=[embedding_matrix], trainable=False)(x)
    shared = Dropout(0.2)(shared)
    shared = Bidirectional(
        LSTM(128, dropout=0.2, return_sequences=True))(shared)
    shared = GlobalMaxPool1D()(shared)
    shared = Dense(128, activation='relu')(shared)

    sub1 = Dense(64, activation='relu')(shared)
    sub1 = Dense(32, activation='relu')(sub1)

    sub2 = Dense(64, activation='relu')(shared)
    sub2 = Dense(32, activation='relu')(sub2)

    sub3 = Dense(64, activation='relu')(shared)
    sub3 = Dense(32, activation='relu')(sub3)

    sub4 = Dense(64, activation='relu')(shared)
    sub4 = Dense(32, activation='relu')(sub4)

    sub5 = Dense(64, activation='relu')(shared)
    sub5 = Dense(32, activation='relu')(sub5)

    out1 = Dense(1, activation='sigmoid')(sub1)
    out2 = Dense(1, activation='sigmoid')(sub2)
    out3 = Dense(1, activation='sigmoid')(sub3)
    out4 = Dense(1, activation='sigmoid')(sub4)
    out5 = Dense(1, activation='sigmoid')(sub5)

    model = Model(inputs=x, outputs=[out1, out2, out3, out4, out5])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[48]:


model = get_model(embedding_matrix, params=params)


# In[49]:


show_model(model)


# In[51]:


def get_callback(params):
    filepath="model/多分类-multi-task/bi-lstm/checkpoint/best.hdf5"
    init_dir(filepath)
    early_stopping_monitor = EarlyStopping(patience=5,verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',period=1)
    return [early_stopping_monitor,checkpoint],filepath


# In[52]:


def save_model(model,params):
    which = params['which']
    which_emb = params['which_emb']
    model_path = 'model/多分类-multi-task/bi-lstm/baseline/bi-lstm_{}_{}.model'.format(which,which_emb)
    init_dir(model_path)
    with open(model_path,'wb') as f:
        pickle.dump(model,f)


# In[53]:


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

# In[54]:


df_train,df_dev,df_test = load_data()


# In[55]:


params = {"max_len": 128, 
          "embed_size": 200,
          "num_labels":5,
          "which_emb": 'tencent', 
          "seg_column":"seg_text",
          "epochs": 100, 
          "which":"002",
          "batch_size": 32}


# In[57]:


x_train, y_train, x_dev, y_dev, x_test, y_test,embedding_matrix = process_feature(
    df_train, df_dev, df_test, params=params)


# In[58]:


# tokenizer = params['tokenizer']
# token_path = 'model/多分类-multi-task/bi-lstm/baseline/tokenizer.plk'
# init_dir(token_path)
# with open(token_path,'wb') as f:
#     pickle.dump(tokenizer,f)


# In[83]:


y_train


# In[59]:


model = train_model(x_train, y_train,x_dev, y_dev,embedding_matrix, params)


# In[ ]:


2.10163


# In[82]:


save_model(model,params)


# ## 评价结果

# In[61]:


y_pred = model.predict(x_test)


# In[75]:


y_pred = np.array(y_pred).reshape(5,-1).argmax(axis=0)


# In[79]:


y_true = np.array(y_test).reshape(5,-1).argmax(axis=0)


# In[80]:


target_names = ['认知问句', '评估问句','自问自答问句', '其它问句', '非提问']
labels = [que_label_map[x] for x in target_names]
print(classification_report(y_true,y_pred,labels=labels,target_names=target_names))


# In[81]:


report = classification_report(y_true,y_pred,labels=labels,target_names=target_names,output_dict=True)
pd.DataFrame(report).T[['precision','recall','f1-score','support']].ix[target_names+['micro avg','macro avg','weighted avg']]


# In[ ]:




