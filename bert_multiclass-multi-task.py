#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/share/small_project/问句识别/为了论文/model/二分类/bert/bert4keras-master/bert4keras-master


# In[1]:


import os
import sys
sys.path.append('/share/tools/tabchen_utils')
from tabchen_utils import *
from sklearn.metrics import classification_report,confusion_matrix
def print_metris(report):
    columns = ['Accuracy','Precision','Recall','F_meansure','AUC_Value']
    print('\t'.join([str(report[x]) for x in columns]))


# In[2]:


from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
def show_model(model):
    return SVG(model_to_dot(model,show_shapes=True,show_layer_names=False).create(prog='dot', format='svg'))


# ## 参考文档
# [bert4keras](https://github.com/bojone/bert4keras)

# In[3]:


import os
import json
import numpy as np
import codecs
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import *
import pandas as pd
set_gelu('tanh')  # 切换gelu版本


# In[4]:


print(keras.__version__)


# In[5]:


def init_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[6]:


maxlen = 128
batch_size = 32
bert_dropout = 0.1
model_dir = 'model/二分类/bert/NEZHA-Base'
config_path = os.path.join(model_dir,'bert_config.json')
checkpoint_path = os.path.join(model_dir,'model.ckpt-900000')
dict_path = os.path.join(model_dir,'vocab.txt')

que_label_map = {'其它问句': 0, '自问自答问句': 1, '认知问句': 2, '评估问句': 3, '非提问': 4}

def load_data(path):
    df = pd.read_csv(path)
    df['label'] = df['question_label'].map(que_label_map)
    D = []
    for text,label in zip(df['text_clean'],df['label']):
        D.append((text, int(label)))
    return D

# 加载数据集
train_data = load_data('data/多分类/train_raw.csv')
valid_data = load_data('data/多分类/dev_raw.csv')
test_data = load_data('data/多分类/test_raw.csv')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# In[24]:


best_model_path = 'model/多分类-multi-task/bert/final_model/NEZHA-Base-实验三_dropout:{}_best_model.weights'.format(bert_dropout)
init_dir(best_model_path)


# In[25]:


# class data_generator(DataGenerator):
#     """数据生成器
#     """
#     def __iter__(self, random=False):
#         idxs = list(range(len(self.data)))
#         if random:
#             np.random.shuffle(idxs)
#         batch_token_ids, batch_segment_ids, batch_labels = [], [], []
#         for i in idxs:
#             text, label = self.data[i]
#             token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
#             batch_token_ids.append(token_ids)
#             batch_segment_ids.append(segment_ids)
#             batch_labels.append([label])
#             if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
#                 batch_token_ids = sequence_padding(batch_token_ids)
#                 batch_segment_ids = sequence_padding(batch_segment_ids)
#                 batch_labels = sequence_padding(batch_labels)
#                 yield [batch_token_ids, batch_segment_ids], convert_multi_label(to_categorical(batch_labels, 5))
#                 batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# In[7]:


def get_feature(data):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in data:
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])
    
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    batch_labels = sequence_padding(batch_labels)
    batch_labels = convert_multi_label(to_categorical(batch_labels, 5))
    return [batch_token_ids, batch_segment_ids],batch_labels


# In[8]:


def convert_multi_label(y):
    return [y[:,i] for i in range(5)]


# In[9]:


# 转换数据集
x_train, y_train = get_feature(train_data)
x_dev, y_dev = get_feature(valid_data)
x_test, y_test = get_feature(test_data)
# test_data_small = copy.deepcopy(test_data)[:200]
# x_test_small,y_test_small = get_feature(test_data_small)


# In[10]:


x_train[0]


# In[11]:


def get_model_3():
    # 加载预训练模型
    bert = build_bert_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='nezha',
        with_pool=True,
        return_keras_model=False,
    )

    shared = Dropout(rate=bert_dropout)(bert.model.output)
#     shared = Dense(512, activation='relu')(shared)
    
    sub1 = Dense(256, activation='relu')(shared)
    sub1 = Dense(64, activation='relu')(sub1)
    out1 = Dense(1, activation='sigmoid')(sub1)

    sub2 = Dense(256, activation='relu')(shared)
    sub2 = Dense(64, activation='relu')(sub2)
    out2 = Dense(1, activation='sigmoid')(sub2)

    sub3 = Dense(256, activation='relu')(shared)
    sub3 = Dense(64, activation='relu')(sub3)
    out3 = Dense(1, activation='sigmoid')(sub3)

    sub4 = Dense(256, activation='relu')(shared)
    sub4 = Dense(64, activation='relu')(sub4)
    out4 = Dense(1, activation='sigmoid')(sub4)

    sub5 = Dense(256, activation='relu')(shared)
    sub5 = Dense(64, activation='relu')(sub5)
    out5 = Dense(1, activation='sigmoid')(sub5)

    model = keras.models.Model(bert.model.input, [out1, out2, out3, out4, out5])
    model.summary()
    return model


# In[30]:


# def get_model_4():
#     # 加载预训练模型
#     bert = build_bert_model(
#         config_path=config_path,
#         checkpoint_path=checkpoint_path,
#         model='nezha',
#         with_pool=True,
#         return_keras_model=False,
#     )

#     shared = Dropout(rate=bert_dropout)(bert.model.output)
#     shared = Dense(512, activation='relu')(shared)
#     shared = Dense(256, activation='relu')(shared)
#     shared = Dropout(rate=0.5)(shared)
    
#     sub1 = Dense(256, activation='relu')(shared)
#     sub1 = Dense(64, activation='relu')(sub1)
#     out1 = Dense(1, activation='sigmoid')(sub1)

#     sub2 = Dense(256, activation='relu')(shared)
#     sub2 = Dense(64, activation='relu')(sub2)
#     out2 = Dense(1, activation='sigmoid')(sub2)

#     sub3 = Dense(256, activation='relu')(shared)
#     sub3 = Dense(64, activation='relu')(sub3)
#     out3 = Dense(1, activation='sigmoid')(sub3)

#     sub4 = Dense(256, activation='relu')(shared)
#     sub4 = Dense(64, activation='relu')(sub4)
#     out4 = Dense(1, activation='sigmoid')(sub4)

#     sub5 = Dense(256, activation='relu')(shared)
#     sub5 = Dense(64, activation='relu')(sub5)
#     out5 = Dense(1, activation='sigmoid')(sub5)

#     model = keras.models.Model(bert.model.input, [out1, out2, out3, out4, out5])
#     model.summary()
#     return model


# In[12]:


model = get_model_3()


# In[14]:


best_nezha_path = '/share/small_project/问句识别/为了论文/model/多分类-multi-task/bert/final_model/NEZHA-Base-实验三_dropout:0.1_best_model.weights'
model.load_weights(best_nezha_path)


# In[17]:


# x_test_2 = [np.array([x_test_small[0][0]]),np.array([x_test_small[1][0]])]


# In[ ]:


show_model(model)


# In[ ]:


AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='binary_crossentropy',
#     optimizer='rmsprop',
#     optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(lr=1e-4,
                     lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_pred = np.array(y_pred).reshape(5,-1).argmax(axis=0)
        y_true = np.array(y_true).reshape(5,-1).argmax(axis=0)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(best_model_path)
        test_acc = evaluate(test_generator)
        print(u'val_acc: %05f, best_val_acc: %05f, test_acc: %05f\n' %
              (val_acc, self.best_val_acc, test_acc))


# In[ ]:


from keras.callbacks import EarlyStopping,ModelCheckpoint
def get_callback():
    init_dir(best_model_path)
    early_stopping_monitor = EarlyStopping(patience=5,verbose=1)
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',period=1)
    return [early_stopping_monitor,checkpoint]


# In[ ]:


callbacks = get_callback()


# In[ ]:


model.fit(x_train, y_train,
          batch_size=16,
          epochs=100,
          validation_data=(x_dev, y_dev), callbacks=callbacks)


# In[17]:


1.01302


# In[ ]:


# evaluator = Evaluator()
# model.fit_generator(train_generator.forfit(),
#                     steps_per_epoch=20,
#                     epochs=20)


# In[37]:


model.load_weights(best_model_path)
# print(u'final test acc: %05f\n' % (evaluate(valid_generator)))


# In[38]:


best_model_path


# ## 评价结果

# In[23]:


# y_pred_list = []
# y_true_list = []
# for x_true, y_test in test_generator:
#     y_pred = model.predict(x_true)
#     y_pred = np.array(y_pred).reshape(5,-1).argmax(axis=0)
#     y_pred_list.extend(list(y_pred))
#     y_true = np.array(y_test).reshape(5,-1).argmax(axis=0)
#     y_true_list.extend(list(y_true))
# y_pred = np.array(y_pred_list) 
# y_true = np.array(y_true_list)


# In[39]:


y_pred = model.predict(x_test)
y_pred = np.array(y_pred).reshape(5,-1).argmax(axis=0)
y_true = np.array(y_test).reshape(5,-1).argmax(axis=0)


# In[40]:


target_names = ['认知问句', '评估问句','自问自答问句', '其它问句', '非提问']
labels = [que_label_map[x] for x in target_names]
print(classification_report(y_true,y_pred,labels=labels,target_names=target_names))


# In[41]:


report = classification_report(y_true,y_pred,labels=labels,target_names=target_names,output_dict=True)
pd.DataFrame(report).T[['precision','recall','f1-score','support']].ix[target_names+['micro avg','macro avg','weighted avg']]


# In[ ]:


cm = confusion_matrix(y_true,y_pred,labels=labels)


# In[ ]:


import numpy as np

sns.set(font="SimHei",font_scale=1.5,context='notebook', style='dark')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('test.jpg')
    plt.show()
    


# In[ ]:


plot_confusion_matrix(cm = cm, 
                      normalize = False,
                      target_names = target_names,
                      title = "Confusion Matrix")


# In[ ]:




