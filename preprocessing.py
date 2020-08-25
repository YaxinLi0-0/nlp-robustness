#!/usr/bin/env python
# coding: utf-8

# In[38]:


import re
import jieba
import pandas as pd
import os


# In[2]:

import ipdb
df = pd.read_csv('train_raw.csv')


# In[3]:


print(df.columns)


# In[4]:


print(df['question_label'].value_counts())


# In[5]:


print(df[df['question_label']!='无人声'].shape)


# In[6]:


df = df[df['question_label']!='无人声']
print(df.shape)


# In[7]:


df_pivot = pd.pivot_table(df[['学部', 'subject_name', 'course_id']], index=[
                          "学部"], columns=['subject_name'],aggfunc=['count'])
#print(df_pivot.ix[['小学','初中','高中']])


# In[4]:


re_q = re.compile("\?|？")
def clean_text(text):
    text = re_q.sub("",text)
    return text


# In[8]:


clean_text("我们?")


# In[9]:


df['text_clean'] = df['text'].apply(clean_text)
df['seg_text'] = df['text_clean'].apply(lambda x:" ".join(jieba.lcut(x)))


# In[10]:


df = df.sample(frac=1,random_state=2019)


# In[11]:


num = df.shape[0]
df_train =  df[:int(num*0.8)]
df_dev = df[int(num*0.8):int(num*0.9)]
df_test = df[int(num*0.9):]


# In[12]:


df_train['question_label'].value_counts(normalize=True)


# In[13]:


df_dev['question_label'].value_counts(normalize=True)


# In[14]:


df_test['question_label'].value_counts(normalize=True)


# In[16]:


df_train.to_csv('data/二分类/train_raw.csv',index=False)
df_dev.to_csv('data/二分类/dev_raw.csv',index=False)
df_test.to_csv('data/二分类/test_raw.csv',index=False)


# In[17]:


df_train['have_question'].value_counts()


# In[8]:


21301 + 10149


# In[18]:


df_dev['have_question'].value_counts()


# In[9]:


2724 + 1207


# In[19]:


df_test['have_question'].value_counts()


# In[11]:


2694 + 1238


# In[ ]:


def load_data():
    root_dir = 'data/多分类/{}'
    df_train = process_df(root_dir.format('train_raw.csv'))
    df_dev = process_df(root_dir.format('dev_raw.csv'))
    df_test = process_df(root_dir.format('test_raw.csv'))
    return df_train,df_dev,df_test


# ## 二分类给问答

# In[3]:


df = pd.read_excel('标注数据/标注结果/全量4w结果.xlsx')


# In[5]:


df = df[df['question_label']!='无人声']
df.shape


# In[6]:


df['text_clean'] = df['text'].apply(clean_text)
df['seg_text'] = df['text_clean'].apply(lambda x:" ".join(jieba.lcut(x)))


# In[39]:


def init_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[44]:


def train_dev_test_split(df, train_size=0.8):
    df = df.sample(frac=1, random_state=0).copy()
    if train_size < 1:
        train_size = int(train_size*df.shape[0])
    num = df.shape[0]
    dev_size = (num-train_size)//2
    df_train = df[:train_size]
    df_dev = df[train_size:dev_size+train_size]
    df_test = df[dev_size+train_size:]
    return df_train, df_dev, df_test


# In[84]:


def split_3_save_data(save_dir,df,train_size=0.8):
    '''split data to train,dev,test.Than save data to savedir.Train_size can be int or float in (0,1)。
    '''
    df_train,df_dev,df_test = train_dev_test_split(df,train_size)
    init_dir(save_dir)
    df_train.to_csv(os.path.join(save_dir,"train.csv"),index=False)
    df_dev.to_csv(os.path.join(save_dir,"dev.csv"),index=False)
    df_test.to_csv(os.path.join(save_dir,"test.csv"),index=False)
    return df_train, df_dev, df_test


# In[58]:


que_list = ['评估问句',"认知问句"]


# In[59]:


df['label'] = df['question_label'].apply(lambda x:1 if x in que_list else 0)


# In[75]:


df = df.drop_duplicates('text')


# In[85]:


save_dir = 'data/二分类给精彩问答/all_data'
df_train, df_dev, df_test = split_3_save_data(save_dir,df)


# In[86]:


df['票数'].value_counts()


# In[87]:


df[df['票数']>=3]['label'].value_counts()


# In[88]:


df[df['票数']>=4]['label'].value_counts()


# In[90]:


df3 = df[df['票数']>=3].copy()


# In[92]:


save_dir = 'data/二分类给精彩问答/3票以上'
df_train, df_dev, df_test = split_3_save_data(save_dir,df3)


# In[94]:


df_train.shape,df_dev.shape,df_test.shape


# In[96]:


df_train['label'].value_counts()


# In[ ]:


#正/负 = 8906/19227


# In[95]:


df_test['label'].value_counts()


# In[93]:


#正/负 = 1172/2345


# In[ ]:


#share(/small_project/问句识别/为了论文/data/二分类给精彩问答/3票以上)

