#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append('/tabchen_utils')
from tabchen_utils import *
import pandas as pd

# In[2]:


def print_metris(report):
    columns = ['Accuracy','Precision','Recall','F_meansure','AUC_Value']
    print('\t'.join([str(report[x]) for x in columns]))


# ## fasttext

# In[3]:


from fasttext.FastText import train_supervised


# In[ ]:


def df_2_ft(df,output_path):
    label_list = df['label'].tolist()
    text_list = df['text_clean'].tolist()
    with open(output_path,'w',encoding='utf-8') as f:
        for label,text in zip(label_list,text_list):
            text = str(text)
            line = '__label__{}\t{}'.format(label,' '.join(jieba.lcut(text)))
            f.write(line+'\n')


# In[7]:


def process_df(path):
    df = pd.read_csv(path)
    df['label'] = df['have_question']
    return df

def load_data():
    root_dir = './{}'
    df_train = process_df(root_dir.format('train_raw.csv'))
    df_dev = process_df(root_dir.format('dev_raw.csv'))
    df_test = process_df(root_dir.format('test_raw.csv'))
    return df_train,df_dev,df_test


# In[8]:


df_train,df_dev,df_test = load_data()


# In[20]:


# df_test['seg_text'] = df_test['text_clean'].apply(lambda x:' '.join(jieba.lcut(str(x))))


# In[9]:


ft_train_path = 'model/二分类/fasttext/ft_train.txt'
ft_test_path = 'model/二分类/fasttext/ft_test.txt'
df_2_ft(df_train,ft_train_path)
df_2_ft(df_test,ft_test_path)


# In[22]:


get_ipython().system('cat 新的标注/model/fasttext/ft_train.txt |wc')


# In[43]:


dim = 200
lr = 0.1
epoch = 100
classifier = train_supervised(ft_train_path, label='__label__', dim=dim, epoch=epoch,
                                         lr=lr, wordNgrams=3, loss='softmax',verbose=2)


# In[44]:


y_pred = np.array([int(x[0].replace('__label__','')) for x in classifier.predict(df_dev['seg_text'].tolist())[0]])


# In[45]:


report = getModelInfo(df_dev['label'],y_pred)
print(report)
print_metris(report)


# In[ ]:


y_pred = np.array([int(x[0].replace('__label__','')) for x in classifier.predict(df_test['seg_text'].tolist())[0]])


# In[ ]:


report = getModelInfo(df_test['label'],y_pred)
print(report)
print_metris(report)

