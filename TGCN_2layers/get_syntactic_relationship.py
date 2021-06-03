#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pickle
import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords


nlp = StanfordCoreNLP(r'.\stanford-corenlp-full-2016-10-31', lang='en')


#路径设置
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))
#dataset: 20ng  mr  ohsumed R8 R52
dataset ='mr'
input = os.sep.join(['..', 'data_tgcn', dataset, 'build_train', dataset])
output = os.sep.join(['..', 'data_tgcn', dataset, 'stanford'])


#读取病例
yic_content_list = []
f = open(input + '.clean.txt', 'r', encoding="gbk")
lines = f.readlines()
for line in lines:
    yic_content_list.append(line.strip())
f.close()

stop_words = set(stopwords.words('english'))

#获取句法依存关系对
rela_pair_count_str = {}
for doc_id in range(len(yic_content_list)):
    print(doc_id)
    words = yic_content_list[doc_id]
    words = words.split("\n")
    rela=[]
    for window in words:
        if window==' ':
            continue
        #构造rela_pair_count
        window = window.replace(string.punctuation, ' ')
        res = nlp.dependency_parse(window)
        for tuple in res:
            rela.append(tuple[0] + ', ' + tuple[1])
        for pair in rela:
            pair=pair.split(", ")
            if pair[0]=='ROOT' or pair[1]=='ROOT':
                continue
            if pair[0] == pair[1]:
                continue
            if pair[0] in string.punctuation or pair[1] in string.punctuation:
                continue
            if pair[0] in stop_words or pair[1] in stop_words:
                continue
            word_pair_str = pair[0] + ',' + pair[1]
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1
            # two orders
            word_pair_str = pair[1] + ',' + pair[0]
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1


#将rela_pair_count_str存成pkl格式
output1=open(output + '/{}_stan.pkl'.format(dataset),'wb')
pickle.dump(rela_pair_count_str, output1)