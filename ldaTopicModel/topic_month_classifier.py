import lda
import datetime
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import time
# 该文件涉及内容多，不能让程序长时间跑该代码，需要将部分内容分离出去，以免出现bug时前功尽弃

'''
不小心改动过index，之前觉得对应关系可能不太对，现在直接就直接用id与文章内容一一对应了
主要是将变量内容通过pickle模块进行保存
'''


# 创建停用词list(可以直接调用黑名单)
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist('../config/stopwords.txt')

# 顺带再按日期排序一下(后续可能会使用)
def month_classifier(doc_keywords_file):
    month_docs_dict = {}
    corpus = list()
    corpus_id = list()
    # 其实也就存在12种情况而已, 如果10几万条数据信息都判断一遍运算效率就大大降低了
    for month in range(12):
        month_docs_dict[month] = list()

    with open(doc_keywords_file, 'r') as fp:
        line = fp.readline().strip()
        while line:
            id, date, kws = line.split(',')

            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            month = date.month - 1 # 月份共分为 0 - 11

            kws = kws.split('|')
            # kws <= 4 的报道存在很多无用内容，根本不需要拿进来一起聚类，直接在这里就把它丢掉
            # corpus_id.append(id) # 所有id
            # corpus.append(' '.join(kws)) # 所有id的kws
            # 调了一下约束，先把数字项给删了，再来判断这篇文章的关键词数有无满足
            # 这种思路删不干净
            kwsWithoutNum = []
            for kw in kws:
                if re.compile(r'-?[0-9]\d*').match(kw):
                    continue
                kwsWithoutNum.append(kw)
            # 仅保留关键词数大于4的报道, 因为我手动改过了源文件。所以到3就行
            if len(kwsWithoutNum) > 3:
                month_docs_dict[month].append((id, date))
                corpus_id.append(id)  # 所有id
                corpus.append(' '.join(kwsWithoutNum))  # 所有id的kws

            line = fp.readline().strip()  # 读取下一行

    return month_docs_dict, corpus_id, corpus


# doc_keywords_file = '/home/zhangchj/PycharmProjects/BLMYX01/res/topic_extraction/2017_doc_keywords.csv'
doc_keywords_file = '/home/zhangchj/PycharmProjects/BLMYX01/res/topic_extraction/dedup_mine_2017_doc_kws2.csv'
month_docs_dict, corpus_id, corpus = month_classifier(doc_keywords_file)
print('finish loading month_docs_dict and corpus')


vectorizer = CountVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
print('finish handling vectorizer')

n_topics = 16 # 划分主题量
n_iter = 2000 # 迭代次数

model =lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
model.fit_transform(X)
print('finish training')

#文档-主题（Document-Topic）分布 (该变量记录了每一篇文档隶属于哪个主题)
doc_topic = model.doc_topic_
print("\n shape: {}".format(doc_topic.shape))


topic_word = model.topic_word_
word = vectorizer.get_feature_names()

n = 20 # 显示关键词数量
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(word)[np.argsort(topic_dist)][:-(n + 1):-1]
    print(u'*Topic {},{}'.format(i, ' '.join(topic_words)))


# print(doc_topic.shape[0], len(corpus_id))
# 由于传入到model中进行训练的只有语料库,只能根据month_docs_dict不同列表中各文档id的顺序确定对应的文档了
# 如下内容可归并到函数中
topic_dict = dict()
index_dict = dict() # 用于统计下每篇文章的分类

for category in range(n_topics):
    topic_dict[category] = 0
    index_dict[category] = list()

# doc_topic.shape[0]和len(corpus_id)是一样大的

for i in range(doc_topic.shape[0]):
    category = doc_topic[i].argmax()
    topic_dict[category] += 1
    index_dict[category].append(corpus_id[i])

# 如下代码负责输出各主题下对应文章数及主题权重
for item in topic_dict.items():
    print("Topic: {}, Num: {}, Weight: {:.2f}%".format(item[0], item[1], 100*item[1]/doc_topic.shape[0]))  # 如此运算无法进行四舍五入(算了，这个误差可以忽略)

# 为方便后续按月、按主题进行层次提取, 将变量先通过pickle存储下来
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

var_pickle_path = '../var_dump/index_dict_2017_{}topics_{}Epochs_{}.var2'.format(n_topics, n_iter, int(time.time()))
if save_variable(index_dict,var_pickle_path):
    print('finish')
# 后面再用map函数整合一下id吧()

# 好像改过corpus_id，之前好像是添加index的