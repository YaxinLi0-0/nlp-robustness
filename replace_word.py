import pycorrector
import pandas as pd
import numpy as np
import json
import sys
from pyhanlp import HanLP   # 调入自然语言处理工具包
import random
from tqdm import tqdm

def replace_synwords(content,synwords):
    """
    使用同义词替换content中的关键词
    :param content:  需要进行同义词替换的句子，不是整个样本或者数据集
    :param synwords: 同义词词典
    :return:
    """
    segmentationList = HanLP.segment(content)
    # print(len(segmentationList))
    if len(set(segmentationList)) <= 2:
        keynum = 1
    elif len(segmentationList) > 2 and len(set(segmentationList)) <= 6:
        keynum = 2
    else:
        # keynum = int(len(set(segmentationList))/3)
        keynum = 4
    keywordList = get_keyword(content,keynum)   # 获取关键词
    # print(keywordList)

    segmentationList = [term.word for term in segmentationList]
    replace_word = {}
    #查询content中的关键词在同义词表中的近义词
    for word in keywordList:
        if word in segmentationList:
            for syn in synwords:
                # if word in syn:   # 设计替换规则
                if word == syn[0]:
                    if len(syn) == 1:
                        continue
                    else:
                        # 以最靠近word的词替换word
                        if syn.index(word) == 0:
                            replace_word[word] = (syn[1])
                        else:
                            replace_word[word] = (syn[syn.index(word)-1])
                else:
                    continue
        else:
            continue

    # 替换content中的同义词
    for i in range(len(segmentationList)):
        if segmentationList[i] in replace_word:
            segmentationList[i] = replace_word[segmentationList[i]]
        else:
            continue
    # 将content重新组合成句子
    content_new = "".join(segmentationList)
    # 返回经过替换后的content,即new_content
    return content_new

def get_word_freq(chinese_word_freq_file_path):
    '''
    读取word,frequency ,构建词典
    :param chinese_word_freq_file_path:   中文词频文件
    :return: {"word1":freq1,"word2":freq2}
    '''
    word_freq_vocab = {} # 词频字典,格式为[“word”:freq]
    with open(chinese_word_freq_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            word_freq = line.split(" ")
            if word_freq[0] not in word_freq_vocab:
                word_freq_vocab[word_freq[0]] = int(word_freq[1])  # 添加"word"：freq到词典中
            else:
                pass
    # print("word_freq_vocab", word_freq_vocab["火"])
    return word_freq_vocab

def get_same_pinyin_vocabulary(same_pinyin_file):
    """
    获得具有相同拼音的词表，文件来自https://github.com/shibing624/pycorrector/blob/master/pycorrector/data/same_pinyin.txt
    :param same_pinyin_file:
    :return: {"word1":samepinyin,"word2":samepinyin}
    """
    same_pinyin = {}
    with open(same_pinyin_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.strip("\n")
            split_1 = temp.split('\t')
            word_index = split_1[0]  # 词根
            # word_same_1 = split_1[1]   #同音同调
            # word_same_2 = split_1[2]   #同音异调
            # word_samePinyin = split_1[1]+split_1[2]
            sameWords = ""
            for i in split_1[1:]:  #拼接同音同调和同音异调词表
                sameWords += i
            same_pinyin[word_index] = list(sameWords)   # 将同音同调和同音异调放在同一个list中
            # same_pinyin[word_index] = [list(word_same_1),list(word_same_2)]   # 将同音同调和同音异调放在不同list中
    # 格式[word,freq]
    return same_pinyin

def get_keyword(content,keynum=2):
    """
    获取每个问题中的关键字,关键词的数目由keynum控制
    :param content: 一个句子
    :return:
    """
    keywordList = HanLP.extractKeyword(content,keynum)
    return keywordList

def replace_samePinyin(content,same_pinyin,word_freq_vocab,replace_num=1):
    """
    使用同音字替换content中关键词中，（替换规则为替换掉所有同音字出现频率最高的那个字）
    :param content:  要替换的文本
    :param same_pinyin: 相同拼音词汇表
    :param word_freq_vocab: 汉语字频率表
    :param replace_num: 要替换的数量，这个版本目前只考虑一个content中只替换一个字
    :return: 经过相同拼音替换掉的文本
    """
    segmentationList = HanLP.segment(content)
    word_list_of_content = list(content)
    # print(len(segmentationList))
    if len(set(segmentationList)) <= 2:
        keynum = 1
    elif len(segmentationList) > 2 and len(set(segmentationList)) <= 6:
        keynum = 2
    else:
        # keynum = int(len(set(segmentationList))/3)
        keynum = 4
    keywordList = get_keyword(content,keynum)   # 获取关键词
    key_character = []
    for word in keywordList:   # 提取关键词里的关键字
        key_character += list(word)
    key_character = list(set(key_character))   # 去掉重复的关键字
    key_character = [word for word in key_character if word in same_pinyin]# 先检查关键词中的所有字是否都出现在same_pinyin词汇表中
    word_freq = []
    for i in key_character:   # 统计关键字的频率
        samePinyin_list = same_pinyin[i]   # 获取相同拼音的所有字
        samePinyin_freq = []
        for j in samePinyin_list:
            if j in word_freq_vocab:
                samePinyin_freq.append(word_freq_vocab[j])
            else:
                samePinyin_freq.append(1)
        word_freq.append(samePinyin_list[samePinyin_freq.index(max(samePinyin_freq))])
    freq =[]
    if len(word_freq) != 0:
        for i in word_freq:
            if i in word_freq_vocab:
                freq.append(word_freq_vocab[i])
            else:
                freq.append(1)
        same_pinyin_HighFreq_word = word_freq[freq.index(max(freq))]
        replace_word = key_character[freq.index(max(freq))]
        replace_index = word_list_of_content.index(replace_word)
        word_list_of_content[replace_index] = same_pinyin_HighFreq_word
        new_content =  "".join(word_list_of_content)
        # print("smae_pinyin",same_pinyin["火"])
        return new_content
    else:
        return content
