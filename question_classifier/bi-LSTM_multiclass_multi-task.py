#!/usr/bin/env python
# coding: utf-8

# # 基本处理

# In[1]:


import os
import sys
sys.path.append('/share/tools/tabchen_utils')
from tabchen_utils import *


# In[2]:


from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot


# In[3]:


# def print_metris(report):
#     columns = ['Accuracy','Precision','Recall','F_meansure','AUC_Value']
#     print('\t'.join([str(report[x]) for x in columns]))


# # 深度网络

# In[3]:


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


# In[4]:


def init_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[26]:


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


# In[7]:


df_train['label'].value_counts()


# In[8]:


df_dev['label'].value_counts()


# In[9]:


df_test['label'].value_counts()


# In[10]:


df = pd.concat([df_train,df_dev,df_test])


# In[30]:


# df = df.fillna(0)


# In[31]:


df.shape


# In[37]:


df.columns


# In[32]:


df["subject_name"].unique()


# In[39]:


df[df['学部'].isna()][['grade','学部','subject']]


# In[33]:


df["学部"].unique()


# In[34]:


df['学部'].value_counts()


# In[27]:


df['subject_name'].value_counts().sum()


# In[28]:


df_pivot = pd.pivot_table(df[['学部', 'subject_name', 'course_id']], index=[
                          "学部"], columns=['subject_name'],aggfunc=['count'])
df_pivot.ix[['小学','初中','高中']]


# ## 数据准备

# In[11]:


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


# In[12]:


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
    y_train_2 = to_categorical(y_train, num_labels)
    y_dev_2 = to_categorical(y_dev, num_labels)
    y_test_2 = to_categorical(y_test, num_labels)

    return x_train_padded_seqs, y_train_2, x_dev_padded_seqs, y_dev_2, x_test_padded_seqs, y_test_2, embedding_matrix


# ## bi-LSTM

# In[13]:


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
    x = Dropout(0.2)(x)
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


# In[14]:


def get_callback(params):
    filepath="model/多分类/bi-lstm/checkpoint/best.hdf5"
    init_dir(filepath)
    early_stopping_monitor = EarlyStopping(patience=5,verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',period=1)
    return [early_stopping_monitor,checkpoint],filepath


# In[15]:


def save_model(model,params):
    which = params['which']
    which_emb = params['which_emb']
    model_path = 'model/多分类/bi-lstm/baseline/bi-lstm_{}_{}.model'.format(which,which_emb)
    init_dir(model_path)
    with open(model_path,'wb') as f:
        pickle.dump(model,f)


# In[17]:


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

# In[18]:


df_train,df_dev,df_test = load_data()


# In[19]:


params = {"max_len": 128, 
          "embed_size": 200,
          "num_labels":5,
          "which_emb": 'tencent', 
          "seg_column":"seg_text",
          "epochs": 100, 
          "which":"002",
          "batch_size": 32}


# In[20]:


x_train, y_train, x_dev, y_dev, x_test, y_test,embedding_matrix = process_feature(
    df_train, df_dev, df_test, params=params)


# In[12]:


tokenizer = params['tokenizer']
token_path = 'model/多分类/bi-lstm/baseline/tokenizer.plk'
with open(token_path,'wb') as f:
    pickle.dump(tokenizer,f)


# In[56]:


model = train_model(x_train, y_train,x_dev, y_dev,embedding_matrix, params)


# In[57]:


save_model(model,params)


# ## 评价结果

# In[22]:


with open('model/多分类/bi-lstm/baseline/bi-lstm_002_tencent.model','rb') as f:
    model = pickle.load(f)


# In[32]:


get_ipython().run_line_magic('pinfo', 'model.predict')


# In[ ]:





# In[23]:


y_pred = model.predict(x_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)


# In[24]:


target_names = ['认知问句', '评估问句','自问自答问句', '其它问句', '非提问']
labels = [que_label_map[x] for x in target_names]
print(classification_report(y_true,y_pred,labels=labels,target_names=target_names))


# In[25]:


report = classification_report(y_true,y_pred,labels=labels,target_names=target_names,output_dict=True)
pd.DataFrame(report).T[['precision','recall','f1-score','support']].ix[target_names+['micro avg','macro avg','weighted avg']]


# ## demo

# In[4]:


with open('model/多分类/bi-lstm/baseline/bi-lstm_002_tencent.model','rb') as f:
    model = pickle.load(f)


# In[5]:


token_path = 'model/多分类/bi-lstm/baseline/tokenizer.plk'
with open(token_path,'rb') as f:
    tokenizer = pickle.load(f)


# In[7]:


def demo(text,model,tokenizer):
    label2name = {0: '其它问句', 1: '自问自答问句', 2: '认知问句', 3: '评估问句', 4: '非提问'}
    words = " ".join(jieba.lcut(text))
    max_len = 128
    tokens = tokenizer.texts_to_sequences([words])
    tokens_padded_seqs = pad_sequences(tokens, maxlen=max_len)
    label = model.predict(tokens_padded_seqs).argmax()
    return {"label":label,"label_name":label2name[label]}


# In[10]:


text = "它要写的是每一个汽化美的主要成分是啥来着"


# In[20]:


words = " ".join(jieba.lcut(text))
max_len = 128
tokens = tokenizer.texts_to_sequences([words])
tokens = tokens*20
tokens_padded_seqs = pad_sequences(tokens, maxlen=max_len)
label = model.predict(tokens_padded_seqs)


# In[24]:


label.argmax(axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


text_list = ['这个题里面，它要写的是每一个汽化美的主要成分是啥来着？',
 '嗯，是单纯的加起来，我是不得注意一下它给的这些数都是啥？',
 '嗯，这个题的这个某一般分析的是为什么这个反应能够自发进行，为什么会发生？',
 '啊A选项正确现在B和C里边还有一个是对的看哪个？',
 '嗯我看A选项为什么选A正反应速率增大后减小？',
 '那为什么会出现先增大的一段是在这一瞬间，你上到了温度，如果升高温度速率是不是瞬间增大？',
 '你这个考虑体积了吗？',
 '再判断的呢？',
 '你先写一下这个溶液的明天的时候，有了我们先写一下质子守恒是啥呀？',
 '那么碳酸氢根水解产物是谁？',
 '那是小好吗？',
 '嗯电电离常数和平衡常数是为了让我们判断电离和谁？',
 '嗯C是错的吗？看一下。',
 '如果把碳酸根消去，我看我能变成C选项，你看啥呀？',
 '对啊质子守恒写完了那来看一下这个选项，我们看选项来写一下物料守恒是啥呀？',
 '那好，那你来看B选项哪个你怎么办？',
 '为什么因为碳酸氢钠解题？',
 '那为什么错了？',
 '所以碳酸氢根的浓度是多少？',
 '如何判断沉淀是否会形成？',
 '我是不是正的溶液中碳酸根母？',
 '这个听懂吗？这块？']


# In[73]:


text = "还有什么力"
demo(text,model,tokenizer)


# In[67]:


# %%time
label_list = []
for text in text_list:
    label = demo(text,model,tokenizer)
    label.update({"text":text})
    label_list.append(label)


# In[62]:


df_label = pd.DataFrame(label_list)


# In[69]:


df_label

