#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# ## 参考文档
# [bert4keras](https://github.com/bojone/bert4keras)

# In[2]:


import os
import json
import numpy as np
import codecs
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import *
import pandas as pd
set_gelu('tanh')  # 切换gelu版本


# In[3]:


def init_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# In[4]:


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


# In[5]:


best_model_path = 'model/多分类/bert/final_model/NEZHA-Base-DNN_dropout:{}_best_model.weights'.format(bert_dropout)
init_dir(best_model_path)


# In[6]:


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# In[7]:


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


# In[8]:


# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=bert_dropout)(bert.model.output)
output = Dense(128, activation='relu')(output)
output = Dense(64, activation="relu")(output)
output = Dense(32, activation="relu")(output)
output = Dense(units=5,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


# In[9]:


AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
#     optimizer=AdamLR(learning_rate=1e-4,
#                      lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
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


# In[10]:


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])


# In[12]:


model.load_weights(best_model_path)
print(u'final test acc: %05f\n' % (evaluate(valid_generator)))


# ## 评价结果

# In[13]:


y_pred_list = []
y_true_list = []
for x_true, y_true in test_generator:
    y_pred = model.predict(x_true).argmax(axis=1)
    y_pred_list.extend(list(y_pred))
    y_true_list.extend(list(y_true.flatten()))


# In[14]:


y_pred = np.array(y_pred_list) 
y_true = np.array(y_true_list)

target_names = ['认知问句', '评估问句','自问自答问句', '其它问句', '非提问']
labels = [que_label_map[x] for x in target_names]
print(classification_report(y_true,y_pred,labels=labels,target_names=target_names))


# In[15]:


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




