#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import numpy as np
import pandas as pd
import sklearn
import pickle
print(sklearn.__version__)
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[2]:


def load_pickle(path):
    with open(path, 'rb') as f:
        pic = pickle.load(f)
    return pic
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# In[3]:


def text2vec_mean(w2v, seg_text, embed_size=200):
    word_list = seg_text.split()
    all_vec = np.array([w2v[x] if x in w2v else np.zeros(embed_size)for x in word_list])
    return np.mean(all_vec, axis=0)

def creat_textvec_df(w2v, df, text_col, label_col):
    all_vec = []
    for idx,row in df.iterrows():
        mean_vec = text2vec_mean(w2v, row[text_col])
        all_vec.append(mean_vec)
    all_vec = np.array(all_vec)
    new_df = pd.DataFrame(all_vec)
    new_df['label'] = df[label_col]
    new_df['seg_text'] = df[text_col]
    print(new_df.shape)
    return new_df


# # 二分类

# ## base functions

# In[4]:


def get_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    print('accuracy={}, precision={}, recall={}, f1={}, auc={}'.format(
        acc, precision, recall, f1, auc
    ))
    
def get_metrics_noauc(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print('accuracy={}, precision={}, recall={}, f1={}'.format(
        acc, precision, recall, f1
    ))


# In[5]:


def LR(X_train, y_train, X_dev, y_dev, X_test, y_test, save_path, n_jobs=10):
    penalty = ['l1', 'l2']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_acc = 0
    best_model = None
    best_acc_tr = 0
    for p in penalty:
        for c in C:
            lr = LogisticRegression(class_weight='balanced', random_state=0,
                                    penalty=p, C=c, max_iter=500, solver='liblinear',
                                   n_jobs=n_jobs)
            lr.fit(X_train, y_train)
            y_pred_tr = lr.predict(X_train)
            y_pred_dev = lr.predict(X_dev)
            acc_tr = accuracy_score(y_train, y_pred_tr)
            acc_dev = accuracy_score(y_dev, y_pred_dev)
            
            if acc_dev > best_acc:
                best_acc = acc_dev
                best_model = lr
                best_acc_tr = acc_tr
    print('train positive {}, negative {}'.format(len([x for x in y_train if x==1]),
                                                 len([x for x in y_train if x==0])))
    print('dev positive {}, negative {}'.format(len([x for x in y_dev if x==1]),
                                                 len([x for x in y_dev if x==0])))
    print('Best Model')
    print(best_model)
    print('train accuracy is {}'.format(best_acc_tr))
    print('dev metrics:')
    get_metrics(best_model, X_dev, y_dev)
    print('test metircs:')
    get_metrics(best_model, X_test, y_test)
    save_pickle(best_model, save_path)
    return best_model


# In[79]:


def GBDT(X_train, y_train, X_dev, y_dev, X_test, y_test, save_path):
    n_estimators = [30,50,100,200] 
    learning_rate = [0.001,0.01,0.1,1] 
    subsample = [0.5,0.7,0.9,1.0] 
    best_acc = 0
    best_model = None
    best_acc_tr = 0
    for n in n_estimators:
        for l in learning_rate:
            for s in subsample:
                gbdt = GradientBoostingClassifier(n_estimators=n, learning_rate=l, 
                                           subsample=s, random_state=0)
                gbdt.fit(X_train, y_train)
                y_pred_tr = gbdt.predict(X_train)
                y_pred_dev = gbdt.predict(X_dev)
                acc_tr = accuracy_score(y_train, y_pred_tr)
                acc_dev = accuracy_score(y_dev, y_pred_dev)

                if acc_dev > best_acc:
                    best_acc = acc_dev
                    best_model = gbdt
                    best_acc_tr = acc_tr
                    
    print('train positive {}, negative {}'.format(len([x for x in y_train if x==1]),
                                                 len([x for x in y_train if x==0])))
    print('dev positive {}, negative {}'.format(len([x for x in y_dev if x==1]),
                                                 len([x for x in y_dev if x==0])))
    print('test positive {}, negative {}'.format(len([x for x in y_test if x==1]),
                                                 len([x for x in y_test if x==0])))
    print('Best Model')
    print(best_model)
    print('train accuracy is {}'.format(best_acc_tr))
    print('dev metrics:')
    get_metrics(best_model, X_dev, y_dev)
    print('test metircs:')
    get_metrics(best_model, X_test, y_test)
    save_pickle(best_model, save_path)
    return best_model


# In[34]:


def SVM(X_train, y_train, X_dev, y_dev, X_test, y_test, save_path):
    kernel = ['linear', 'poly', 'rbf']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_acc = 0
    best_model = None
    best_acc_tr = 0
    for k in kernel:
        for c in C:
            svm = SVC(class_weight='balanced', random_state=0, probability=True, 
                                    kernel=k, C=c)
            svm.fit(X_train, y_train)
            y_pred_tr = svm.predict(X_train)
            y_pred_dev = svm.predict(X_dev)
            acc_tr = accuracy_score(y_train, y_pred_tr)
            acc_dev = accuracy_score(y_dev, y_pred_dev)
            
            if acc_dev > best_acc:
                best_acc = acc_dev
                best_model = svm
                best_acc_tr = acc_tr
    print('train positive {}, negative {}'.format(len([x for x in y_train if x==1]),
                                                 len([x for x in y_train if x==0])))
    print('dev positive {}, negative {}'.format(len([x for x in y_dev if x==1]),
                                                 len([x for x in y_dev if x==0])))
    print('Best Model')
    print(best_model)
    print('train accuracy is {}'.format(best_acc_tr))
    print('dev metrics:')
    get_metrics(best_model, X_dev, y_dev)
    print('test metircs:')
    get_metrics(best_model, X_test, y_test)
    save_pickle(best_model, save_path)
    return best_model


# In[61]:


def random_forest(X_train, y_train, X_dev, y_dev, X_test, y_test, save_path, n_jobs=10):
    n_estimators = [30,50,100,200]
    best_acc = 0
    best_model = None
    best_acc_tr = 0
    for n in n_estimators:
        rf = RandomForestClassifier(class_weight='balanced', random_state=0,
                                n_estimators=n, n_jobs=n_jobs)
        rf.fit(X_train, y_train)
        y_pred_tr = rf.predict(X_train)
        y_pred_dev = rf.predict(X_dev)
        acc_tr = accuracy_score(y_train, y_pred_tr)
        acc_dev = accuracy_score(y_dev, y_pred_dev)

        if acc_dev > best_acc:
            best_acc = acc_dev
            best_model = rf
            best_acc_tr = acc_tr
    print('train positive {}, negative {}'.format(len([x for x in y_train if x==1]),
                                                 len([x for x in y_train if x==0])))
    print('dev positive {}, negative {}'.format(len([x for x in y_dev if x==1]),
                                                 len([x for x in y_dev if x==0])))
    print('Best Model')
    print(best_model)
    print('train accuracy is {}'.format(best_acc_tr))
    print('dev metrics:')
    get_metrics(best_model, X_dev, y_dev)
    print('test metircs:')
    get_metrics(best_model, X_test, y_test)
    save_pickle(best_model, save_path)
    return best_model


# In[24]:


def kneighbor(X_train, y_train, X_dev, y_dev, X_test, y_test, save_path, n_jobs=10):
    n_neighbors = [3,5,7,9,11,13]
    best_acc = 0
    best_model = None
    best_acc_tr = 0
    for n in n_neighbors:
        kn = KNeighborsClassifier(n_neighbors=n, n_jobs=n_jobs)
        kn.fit(X_train, y_train)
        y_pred_tr = kn.predict(X_train)
        y_pred_dev = kn.predict(X_dev)
        acc_tr = accuracy_score(y_train, y_pred_tr)
        acc_dev = accuracy_score(y_dev, y_pred_dev)

        if acc_dev > best_acc:
            best_acc = acc_dev
            best_model = kn
            best_acc_tr = acc_tr
    print('train positive {}, negative {}'.format(len([x for x in y_train if x==1]),
                                                 len([x for x in y_train if x==0])))
    print('dev positive {}, negative {}'.format(len([x for x in y_dev if x==1]),
                                                 len([x for x in y_dev if x==0])))
    print('Best Model')
    print(best_model)
    print('train accuracy is {}'.format(best_acc_tr))
    print('dev metrics:')
    get_metrics(best_model, X_dev, y_dev)
    print('test metircs:')
    get_metrics(best_model, X_test, y_test)
    save_pickle(best_model, save_path)
    return best_model


# In[30]:


def bayes_gaussian(X_train, y_train, X_dev, y_dev, X_test, y_test, save_path):
    smooths = []
    best_acc = 0
    best_model = None
    best_acc_tr = 0
    
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    y_pred_tr = bayes.predict(X_train)
    y_pred_dev = bayes.predict(X_dev)
    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_dev = accuracy_score(y_dev, y_pred_dev)

    if acc_dev > best_acc:
        best_acc = acc_dev
        best_model = bayes
        best_acc_tr = acc_tr
    print('train positive {}, negative {}'.format(len([x for x in y_train if x==1]),
                                                 len([x for x in y_train if x==0])))
    print('dev positive {}, negative {}'.format(len([x for x in y_dev if x==1]),
                                                 len([x for x in y_dev if x==0])))
    print('Best Model')
    print(best_model)
    print('train accuracy is {}'.format(best_acc_tr))
    print('dev metrics:')
    get_metrics(best_model, X_dev, y_dev)
    print('test metircs:')
    get_metrics(best_model, X_test, y_test)
    save_pickle(best_model, save_path)
    return best_model


# In[ ]:





# ## 实验

# In[14]:


# load raw data
data_dir = './data/二分类'
train_raw = pd.read_csv(data_dir+'/train_raw.csv')
dev_raw = pd.read_csv(data_dir+'/dev_raw.csv')
test_raw = pd.read_csv(data_dir+'/test_raw.csv')
print(test_raw.columns)


# In[94]:


all_data = pd.concat([train_raw, dev_raw, test_raw], ignore_index=True)
print(all_data.shape)


# In[96]:


all_data.to_csv(data_dir+'/all_raw.csv', index=False)


# In[105]:


all_data[all_data['subject_name'].isna()]


# In[21]:


model_save_dir = './model/二分类/basemodels/'


# In[40]:


print(test_raw['question_label'].unique())


# In[16]:


train_raw['学部'].unique()


# In[ ]:





# ### 文本特征

# In[17]:


# load word vector
tencent_vec = load_pickle(data_dir+'/tencent_small.plk')
ziyan_vec = load_pickle(data_dir+'/ziyan_small.plk')


# In[18]:


tencent_train = creat_textvec_df(tencent_vec, train_raw, 'seg_text', 'have_question')
tencent_dev = creat_textvec_df(tencent_vec, dev_raw, 'seg_text', 'have_question')
tencent_test = creat_textvec_df(tencent_vec, test_raw, 'seg_text', 'have_question')


# In[19]:


ziyan_train = creat_textvec_df(ziyan_vec, train_raw, 'seg_text', 'have_question')
ziyan_dev = creat_textvec_df(ziyan_vec, dev_raw, 'seg_text', 'have_question')
ziyan_test = creat_textvec_df(ziyan_vec, test_raw, 'seg_text', 'have_question')


# In[20]:


X_tencent_tr = tencent_train.drop(columns=['seg_text','label'],axis=1)
y_tencent_tr = tencent_train['label']
print('X train {}, y train {}'.format(X_tencent_tr.shape, len(y_tencent_tr)))
X_tencent_dev = tencent_dev.drop(columns=['seg_text','label'],axis=1)
y_tencent_dev = tencent_dev['label']
print('X train {}, y train {}'.format(X_tencent_dev.shape, len(y_tencent_dev)))
X_tencent_test = tencent_test.drop(columns=['seg_text','label'],axis=1)
y_tencent_test = tencent_test['label']
print('X train {}, y train {}'.format(X_tencent_test.shape, len(y_tencent_test)))


# In[25]:


# LR
tencent_lr = LR(X_tencent_tr, y_tencent_tr, 
                X_tencent_dev, y_tencent_dev, 
                X_tencent_test, y_tencent_test)


# In[30]:


# gbdt
tencent_gbdt = GBDT(X_tencent_tr, y_tencent_tr, 
                X_tencent_dev, y_tencent_dev, 
                X_tencent_test, y_tencent_test)


# In[33]:


# svm
tencent_svm = SVM(X_tencent_tr, y_tencent_tr, 
                X_tencent_dev, y_tencent_dev, 
                X_tencent_test, y_tencent_test,
                model_save_dir+'svm_tencent.model')


# In[35]:


tencent_svm = SVC(class_weight='balanced', random_state=0, probability=True, 
                  kernel='rbf', C=1000)
tencent_svm.fit(X_tencent_tr, y_tencent_tr)
get_metrics(tencent_svm, X_tencent_test, y_tencent_test)
save_pickle(tencent_svm, model_save_dir+'svm_tencent.model')


# In[22]:


# random forest
tencent_rf = random_forest(X_tencent_tr, y_tencent_tr, 
                X_tencent_dev, y_tencent_dev, 
                X_tencent_test, y_tencent_test,
                model_save_dir+'randomForest_tencent.model')


# In[25]:


# k neighbors
tencent_kn = kneighbor(X_tencent_tr, y_tencent_tr, 
                X_tencent_dev, y_tencent_dev, 
                X_tencent_test, y_tencent_test,
                model_save_dir+'kNeighbors_tencent.model')


# In[31]:


# bayes
tencent_bayes = bayes_gaussian(X_tencent_tr, y_tencent_tr, 
                X_tencent_dev, y_tencent_dev, 
                X_tencent_test, y_tencent_test,
                model_save_dir+'gaubayes_tencent.model')


# ### opensmile特征

# In[37]:


op=pd.read_csv('/share/small_project/问句识别/为了论文/data/总标注数据/opensmile1582.csv')
print(op.shape)


# In[43]:


op = op.rename(columns={'id':'audio_name'})


# In[45]:


op.iloc[0]['audio_name']


# In[ ]:


op_train=train_raw[['audio_name','have_question','']].merge()


# ### 文本+opensmile特征

# In[41]:


tencent_test['audio_name']=test_raw['audio_name'].tolist()
tencent_train['audio_name']=train_raw['audio_name'].tolist()
tencent_dev['audio_name']=dev_raw['audio_name'].tolist()


# # 多分类

# In[37]:


multi_save_dir = './model/多分类/basemodels/'


# In[58]:


def LR_multi(X_train, y_train, X_dev, y_dev, X_test, y_test, classes, save_path, n_jobs=10):
    penalty = ['l2']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_f1 = 0
    best_model = None
    best_f1_tr = 0
    for p in penalty:
        for c in C:
            lr = LogisticRegression(class_weight='balanced', random_state=0,
                                    penalty=p, C=c, max_iter=300, solver='lbfgs',
                                   n_jobs=n_jobs, multi_class='auto')
            lr.fit(X_train, y_train)
            y_pred_tr = lr.predict(X_train)
            y_pred_dev = lr.predict(X_dev)
            f1_tr = f1_score(y_train, y_pred_tr, average='micro')
            f1_dev = f1_score(y_dev, y_pred_dev, average='micro')    
            if f1_dev > best_f1:
                best_f1 = f1_dev
                best_model = lr
                best_f1_tr = f1_tr
    y_pred_dev = best_model.predict(X_dev)
    y_pred_test = best_model.predict(X_test)
    print('Best Model')
    print(best_model)
    print('train f1 is {}'.format(best_f1_tr))
    print('dev metrics:')
    print(classification_report(y_dev, y_pred_dev, target_names=classes))
    print('test metircs:')
    print(classification_report(y_test, y_pred_test, target_names=classes))
    save_pickle(best_model, save_path)
    return best_model


# In[62]:


def random_forest_multi(X_train, y_train, X_dev, y_dev, X_test, y_test, classes, save_path, n_jobs=10):
    n_estimators = [30,50,100,200]
    best_f1 = 0
    best_model = None
    best_f1_tr = 0
    for n in n_estimators:
        rf = RandomForestClassifier(class_weight='balanced', random_state=0,
                                n_estimators=n, n_jobs=n_jobs)
        rf.fit(X_train, y_train)
        y_pred_tr = rf.predict(X_train)
        y_pred_dev = rf.predict(X_dev)
        f1_tr = f1_score(y_train, y_pred_tr, average='micro')
        f1_dev = f1_score(y_dev, y_pred_dev, average='micro')
        if f1_dev > best_f1:
            best_f1 = f1_dev
            best_model = rf
            best_f1_tr = f1_tr
    y_pred_dev = best_model.predict(X_dev)
    y_pred_test = best_model.predict(X_test)
    print('Best Model')
    print(best_model)
    print('train f1 is {}'.format(best_f1_tr))
    print('dev metrics:')
    print(classification_report(y_dev, y_pred_dev, target_names=classes))
    print('test metircs:')
    print(classification_report(y_test, y_pred_test, target_names=classes))
    save_pickle(best_model, save_path)
    return best_model


# In[65]:


def kneighbor_multi(X_train, y_train, X_dev, y_dev, X_test, y_test, classes, save_path, n_jobs=10):
    n_neighbors = [3,5,7,9,11,13]
    best_f1 = 0
    best_model = None
    best_f1_tr = 0
    for n in n_neighbors:
        kn = KNeighborsClassifier(n_neighbors=n, n_jobs=n_jobs)
        kn.fit(X_train, y_train)
        y_pred_tr = kn.predict(X_train)
        y_pred_dev = kn.predict(X_dev)
        f1_tr = f1_score(y_train, y_pred_tr, average='micro')
        f1_dev = f1_score(y_dev, y_pred_dev, average='micro')
        if f1_dev > best_f1:
            best_f1 = f1_dev
            best_model = kn
            best_f1_tr = f1_tr
    y_pred_dev = best_model.predict(X_dev)
    y_pred_test = best_model.predict(X_test)
    
    print('Best Model')
    print(best_model)
    print('train f1 is {}'.format(best_f1_tr))
    print('dev metrics:')
    print(classification_report(y_dev, y_pred_dev, target_names=classes))
    print('test metircs:')
    print(classification_report(y_test, y_pred_test, target_names=classes))
    save_pickle(best_model, save_path)
    return best_model


# In[70]:


def SVM_multi(X_train, y_train, X_dev, y_dev, X_test, y_test, classes, save_path):
    kernel = ['linear', 'poly', 'rbf']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_f1 = 0
    best_model = None
    best_f1_tr = 0
    for k in kernel:
        for c in C:
            svm = SVC(class_weight='balanced', random_state=0, probability=True, 
                                    kernel=k, C=c)
            svm.fit(X_train, y_train)
            y_pred_tr = svm.predict(X_train)
            y_pred_dev = svm.predict(X_dev)
            f1_tr = f1_score(y_train, y_pred_tr, average='micro')
            f1_dev = f1_score(y_dev, y_pred_dev, average='micro')
            
            if f1_dev > best_f1:
                best_f1 = f1_dev
                best_model = svm
                best_f1_tr = f1_tr

    y_pred_dev = best_model.predict(X_dev)
    y_pred_test = best_model.predict(X_test)
    
    print('Best Model')
    print(best_model)
    print('train f1 is {}'.format(best_f1_tr))
    print('dev metrics:')
    print(classification_report(y_dev, y_pred_dev, target_names=classes))
    print('test metircs:')
    print(classification_report(y_test, y_pred_test, target_names=classes))
    save_pickle(best_model, save_path)
    return best_model


# In[82]:


def GBDT_multi(X_train, y_train, X_dev, y_dev, X_test, y_test, classes, save_path):
    n_estimators = [30,50,100,200] 
    learning_rate = [0.001,0.01,0.1,1] 
    subsample = [0.5,0.7,0.9,1.0] 
    best_f1 = 0
    best_model = None
    best_f1_tr = 0
    for n in n_estimators:
        for l in learning_rate:
            for s in subsample:
                gbdt = GradientBoostingClassifier(n_estimators=n, learning_rate=l, 
                                           subsample=s, random_state=0)
                gbdt.fit(X_train, y_train)
                y_pred_tr = gbdt.predict(X_train)
                y_pred_dev = gbdt.predict(X_dev)
                f1_tr = f1_score(y_train, y_pred_tr, average='micro')
                f1_dev = f1_score(y_dev, y_pred_dev, average='micro')

                if f1_dev > best_f1:
                    best_f1 = f1_dev
                    best_model = gbdt
                    best_f1_tr = f1_tr
                     
    y_pred_dev = best_model.predict(X_dev)
    y_pred_test = best_model.predict(X_test)
    
    print('Best Model')
    print(best_model)
    print('train f1 is {}'.format(best_f1_tr))
    print('dev metrics:')
    print(classification_report(y_dev, y_pred_dev, target_names=classes, digits=4))
    print('test metircs:')
    print(classification_report(y_test, y_pred_test, target_names=classes, digits=4))
    save_pickle(best_model, save_path)
    return best_model


# In[44]:


tencent_train['question_label'] = train_raw['question_label'].tolist()
tencent_dev['question_label'] = dev_raw['question_label'].tolist()
tencent_test['question_label'] = test_raw['question_label'].tolist()


# In[45]:


tencent_test['question_label'].unique()


# In[46]:


tencent_train = tencent_train.replace({'非提问':0,'评估问句':1,'认知问句':2,'自问自答问句':3,
                                      '其它问句':4})
tencent_dev = tencent_dev.replace({'非提问':0,'评估问句':1,'认知问句':2,'自问自答问句':3,
                                      '其它问句':4})
tencent_test = tencent_test.replace({'非提问':0,'评估问句':1,'认知问句':2,'自问自答问句':3,
                                      '其它问句':4})


# In[49]:


print(tencent_train.columns)
print(tencent_train['question_label'].unique())


# In[47]:


classes = ['非提问', '评估问句', '认知问句', '自问自答问句', '其它问句']


# In[50]:


X_tencent_tr_multi = tencent_train.drop(columns=['seg_text','label','question_label'],axis=1)
y_tencent_tr_multi = tencent_train['question_label']
print('X train {}, y train {}'.format(X_tencent_tr_multi.shape, len(y_tencent_tr_multi)))
X_tencent_dev_multi = tencent_dev.drop(columns=['seg_text','label','question_label'],axis=1)
y_tencent_dev_multi = tencent_dev['question_label']
print('X train {}, y train {}'.format(X_tencent_dev_multi.shape, len(y_tencent_dev_multi)))
X_tencent_test_multi = tencent_test.drop(columns=['seg_text','label','question_label'],axis=1)
y_tencent_test_multi = tencent_test['question_label']
print('X train {}, y train {}'.format(X_tencent_test_multi.shape, len(y_tencent_test_multi)))


# In[57]:


tencent_lr_multi = LR_multi(X_tencent_tr_multi, y_tencent_tr_multi, 
                X_tencent_dev_multi, y_tencent_dev_multi, 
                X_tencent_test_multi, y_tencent_test_multi, 
                classes, multi_save_dir+'lr_tencent.model')


# In[89]:


tencent_lr_multi = load_pickle(multi_save_dir+'lr_tencent.model')
y_pred_dev = tencent_lr_multi.predict(X_tencent_dev_multi)
y_pred_test = tencent_lr_multi.predict(X_tencent_test_multi)
print('dev metrics:')
print(classification_report(y_tencent_dev_multi, y_pred_dev, target_names=classes, digits=3))
print('test metircs:')
print(classification_report(y_tencent_test_multi, y_pred_test, target_names=classes, digits=3))


# In[64]:


tencent_rf_multi = random_forest_multi(X_tencent_tr_multi, y_tencent_tr_multi, 
                X_tencent_dev_multi, y_tencent_dev_multi, 
                X_tencent_test_multi, y_tencent_test_multi, 
                classes, multi_save_dir+'randomForest_tencent.model')


# In[90]:


tencent_rf_multi = load_pickle(multi_save_dir+'randomForest_tencent.model')
y_pred_dev = tencent_rf_multi.predict(X_tencent_dev_multi)
y_pred_test = tencent_rf_multi.predict(X_tencent_test_multi)
print('dev metrics:')
print(classification_report(y_tencent_dev_multi, y_pred_dev, target_names=classes, digits=3))
print('test metircs:')
print(classification_report(y_tencent_test_multi, y_pred_test, target_names=classes, digits=3))


# In[66]:


tencent_kn_multi = kneighbor_multi(X_tencent_tr_multi, y_tencent_tr_multi, 
                X_tencent_dev_multi, y_tencent_dev_multi, 
                X_tencent_test_multi, y_tencent_test_multi, 
                classes, multi_save_dir+'kNeighbors_tencent.model')


# In[75]:


tencent_svm_multi = SVM_multi(X_tencent_tr_multi, y_tencent_tr_multi, 
                X_tencent_dev_multi, y_tencent_dev_multi, 
                X_tencent_test_multi, y_tencent_test_multi, 
                classes, multi_save_dir+'svm_tencent.model')


# In[93]:


tencent_svm_multi2 = load_pickle(multi_save_dir+'svm_tencent.model')
y_pred_dev = tencent_svm_multi2.predict(X_tencent_dev_multi)
y_pred_test = tencent_svm_multi2.predict(X_tencent_test_multi)
print('dev metrics:')
print(classification_report(y_tencent_dev_multi, y_pred_dev, target_names=classes, digits=3))
print('test metircs:')
print(classification_report(y_tencent_test_multi, y_pred_test, target_names=classes, digits=3))


# In[ ]:


tencent_gbdt_multi = GBDT_multi(X_tencent_tr_multi, y_tencent_tr_multi, 
                X_tencent_dev_multi, y_tencent_dev_multi, 
                X_tencent_test_multi, y_tencent_test_multi, 
                classes, multi_save_dir+'gbdt_tencent.model')


# In[92]:


tencent_gbdt_multi2 = load_pickle(multi_save_dir+'gbdt_tencent.model')
y_pred_dev = tencent_gbdt_multi2.predict(X_tencent_dev_multi)
y_pred_test = tencent_gbdt_multi2.predict(X_tencent_test_multi)
print('dev metrics:')
print(classification_report(y_tencent_dev_multi, y_pred_dev, target_names=classes, digits=3))
print('test metircs:')
print(classification_report(y_tencent_test_multi, y_pred_test, target_names=classes, digits=3))


# In[86]:


tencent_gbdt_multi2


# In[ ]:




