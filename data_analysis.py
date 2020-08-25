#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('标注数据/标注结果/全量4w结果.xlsx')
df.columns


# In[3]:


for subject_name,group in df.groupby('subject_name'):
    for nb,nb_group in group.groupby('学部'):
        print(subject_name,nb,nb_group['course_id'].unique().shape[0])


# In[4]:


df_pivot = pd.pivot_table(df[['学部', 'subject_name', 'course_id']], index=[
                          "学部"], columns=['subject_name'],aggfunc=['count'])
df_pivot.ix[['小学','初中','高中']]


# In[5]:


df['course_id'].unique().shape


# In[6]:


df['text_len'] = df.text.apply(lambda x: len(re.sub(r'[^\w]|_','', str(x))))


# In[7]:


df = df[df['question_label']!='无人声'].copy()
df['question type'] = df['question_label']
df['subject'] = df['subject_name']
df['grade'] = df['学部']


# In[ ]:


df['duration'].descirbe


# In[13]:


df['duration'].mean()


# In[8]:


sns.set(font="SimHei",font_scale=2)


# In[9]:


que_types = ['认知问句', '评估问句','自问自答问句', '其它问句', '非提问']
grade_list = ['小学', '初中','高中']
subject_list = ['数学', '化学','物理']


# In[9]:


def count_plot(x,df,order=[],hue=None,hue_order=[],figsize=(15,8),normalize=False):
    
    plt.figure(figsize=figsize)
    if hue is None and normalize:
        ax = sns.barplot(x=x, y='have_question', hue=hue,data=df, estimator=lambda x: len(x) / len(df),hue_order=hue_order,order=order)
        ax.set(ylabel="Percent")
    elif hue is None and not normalize:
        ax = sns.countplot(x=x, data=df,order=order,hue_order=hue_order,hue=hue)
    else:
        y = "Percent"
        prop_df = (df[x]
           .groupby(df[hue])
           .value_counts(normalize=normalize)
           .rename(y)
           .reset_index())
        print(prop_df)
        sns.barplot(x=x, y=y, hue=hue, data=prop_df,order=order,hue_order=hue_order)


# In[8]:


count_plot(x='question type',df=df,order=que_types)


# In[9]:


count_plot(x='question type',df=df,order=que_types,normalize=True)


# In[10]:


count_plot(x='question type',df=df,order=que_types,hue="subject",hue_order=subject_list)


# In[11]:


count_plot(x='question type',df=df,order=que_types,hue="subject",hue_order=subject_list,normalize=True)


# In[12]:


count_plot(x='question type',df=df,order=que_types,hue="grade",hue_order=grade_list)


# In[13]:


count_plot(x='question type',df=df,order=que_types,hue="grade",hue_order=grade_list,normalize=True)


# In[14]:


count_plot(x='grade',df=df,order=grade_list,hue="question type",hue_order=que_types)


# In[24]:


# count_plot(x='grade',df=df,order=grade_list,hue="question type",hue_order=que_types,normalize=True)


# In[16]:


count_plot(x='have_question',df=df,order=[0,1],hue="subject",hue_order=subject_list)


# In[17]:


count_plot(x='have_question',df=df,order=[0,1],hue="subject",hue_order=subject_list,normalize=True)


# In[18]:


count_plot(x='have_question',df=df,order=[0,1],hue="grade",hue_order=grade_list)


# In[19]:


count_plot(x='have_question',df=df,order=[0,1],hue="grade",hue_order=grade_list,normalize=True)


# In[20]:


count_plot(x='have_question',df=df,order=[0,1])


# In[21]:


df.groupby('question type')['text_len'].describe()


# In[22]:


df_pivot = pd.pivot_table(df[['学部', 'subject_name', 'course_id']], index=[
                          "学部"], columns=['subject_name'],aggfunc=['count'])
df_pivot.ix[['小学','初中','高中']]


# In[25]:


plt.figure(figsize=(15,8))
sns.boxplot(x="question type", y="text_len",data=df,order=que_types,hue='grade',hue_order=grade_list);


# In[ ]:




