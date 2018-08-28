import lda
import datetime
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
'''
A Modified versino of hierarchy_topic_extractor.py
层次主题聚类
'''

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

var_pickle_path = '../var_dump/index_dict_2017_17topics_2000Epochs_1534920610.variable'
index_dict = load_variavle(var_pickle_path)

# 创建停用词list(可以直接调用黑名单)
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist('../config/stopwords.txt')

# 层次抽取之后再看按月的情况如何（这边给过来的文章id都已经是大于4了，在函数里边没必要再判断一次）
# ggkf_corpus_id = index_dict[2]
# party_corpus_id = index_dict[5]
# economy_corpus_id = index_dict[7]
# medicalEdu_corpus_id = index_dict[10]
corpus_id = index_dict[15]


def month_classifier(doc_keywords_file, corpus_id):
    month_docs_dict = {}
    corpus = list()
    corpus_index = list()

    # 其实也就存在12种情况而已, 如果10几万条数据信息都判断一遍运算效率就大大降低了
    for month in range(12):
        month_docs_dict[month] = list()

    with open(doc_keywords_file, 'r') as fp:
        line = fp.readline().strip()
        while line:
            id, date, kws = line.split(',')
            if id in corpus_id:
                kws = kws.split('|')
                date = datetime.datetime.strptime(date, "%Y-%m-%d")
                month = date.month - 1  # 月份共分为 0 - 11
                # 仅保留关键词数大于4的报道
                kwsWithoutNum = []
                for kw in kws:
                    if re.compile(r'-?[0-9]\d*').match(kw):
                        continue
                    kwsWithoutNum.append(kw)

                if len(kwsWithoutNum) > 4:
                    content = ' '.join(kwsWithoutNum)
                    corpus.append(content)
                    corpus_index.append(id)
                    month_docs_dict[month].append((id, date, content))
            line = fp.readline().strip()  # 读取下一行
    return month_docs_dict, corpus, corpus_index

doc_keywords_file = '/home/zhangchj/PycharmProjects/BLMYX01/res/topic_extraction/dedup_mine_2017_doc_kws.csv'
month_docs_dict, corpus, corpus_index = month_classifier(doc_keywords_file, corpus_id)
print('finish loading month_docs_dict and corpus')

report_num = []
for month in range(12):
     report_num.append(len(month_docs_dict[month]))

print(report_num) # 观察随着月份推移相关主题文章数量的变化

# vectorizer = CountVectorizer(stop_words=stopwords)
# X = vectorizer.fit_transform(corpus)
# analyze = vectorizer.build_analyzer()
# print('finish handling vectorizer')
#
# n_topics = 8 # 划分主题量
# n_iter = 500 # 迭代次数
#
# model =lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
# model.fit_transform(X)
# print('finish training')
#
# #文档-主题（Document-Topic）分布 (该变量记录了每一篇文档隶属于哪个主题)
# doc_topic = model.doc_topic_
# print("\n shape: {}".format(doc_topic.shape))
# topic_word = model.topic_word_
# word = vectorizer.get_feature_names()
#
# n = 20 # 显示关键词数量（这个是按頻数)
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(word)[np.argsort(topic_dist)][:-(n + 1):-1]
#     print(u'*Topic {},{}'.format(i, ' '.join(topic_words)))
#
# topic_dict = dict()
# # 先初始化所有类别的索引
# for category in range(n_topics):
#     topic_dict[category] = 0
#
#
# for index in range(len(corpus_index)):
#     category = doc_topic[index].argmax()
#     topic_dict[category] += 1
#
#
# # 如下代码负责输出各主题下对应文章数及主题权重
# for item in topic_dict.items():
#     print("Topic: {}, Num: {}, Weight: {:.2f}%".format(item[0], item[1], 100*item[1]/len(corpus_id)))  # 如此运算无法进行四舍五入(算了，这个误差可以忽略)
#
#


# TODO 写个能够按月份、并且按日期先后进行排序的函数
# e.g. a = [('1000','2017-02-05'), ('1001', '2017-01-01'),('1002', '2017-03-06')]
# sorted(a, key=lambda x:x[1])   ：    [('1001', '2017-01-01'), ('1000', '2017-02-05'), ('1002', '2017-03-06')]




