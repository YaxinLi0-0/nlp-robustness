#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/share/small_project/问句识别/为了论文/model/二分类/bert/bert4keras-master/bert4keras-master


# In[1]:


import os
import sys
sys.path.append('./tabchen_utils')
from tabchen_utils import *
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
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import *
import pandas as pd
set_gelu('tanh')  # 切换gelu版本


# In[ ]:


maxlen = 128
batch_size = 4
model_dir = 'model/二分类/bert/NEZHA-Base'
config_path = os.path.join(model_dir,'bert_config.json')
checkpoint_path = os.path.join(model_dir,'model.ckpt-900000')
dict_path = os.path.join(model_dir,'vocab.txt')


def load_data(path):
    df = pd.read_csv(path)
    D = []
    for text,label in zip(df['text_clean'],df['have_question']):
        D.append((text, int(label)))
    return D

# 加载数据集
train_data = load_data('./train_raw.csv')
valid_data = load_data('./dev_raw.csv')
test_data = load_data('./test_raw.csv')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# In[5]:


best_model_path = 'model/二分类/bert/final_model/NEZHA-Base-DNN_best_model.weights'


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


# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(128, activation='relu')(output)
output = Dense(64, activation="relu")(output)
output = Dense(32, activation="relu")(output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


# In[ ]:


AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
#     optimizer=AdamLR(learning_rate=1e-4,
#                      lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


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


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])


# In[ ]:


model.load_weights(best_model_path)
print(u'final test acc: %05f\n' % (evaluate(test_generator)))


# In[ ]:


y_pred_list = []
y_true_list = []
for x_true, y_true in test_generator:
    y_pred = model.predict(x_true)[:,1]
    y_pred_list.extend(list(y_pred))
    y_true_list.extend(list(y_true.flatten()))


# In[ ]:


y_pred = np.array(y_pred_list) 
y_test = np.array(y_true_list)
print_metris(getModelInfo(y_test,y_pred))


# In[ ]:




