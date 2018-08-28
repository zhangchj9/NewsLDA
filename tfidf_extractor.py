# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle # 以防万一
import re

'''
从文章内容的结构化抽取（关键词提取）入手从头开始
这个list的过程太漫长了(3小时内无法解决)，得考虑先把数据文件进行拆分（用shell吧）
'''

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 为方便后续按月、按主题进行层次提取, 将变量先通过pickle存储下来
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


# 创建停用词list(可以直接调用黑名单)
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  # 已经确保不含有重复内容
    stopwords = set(stopwords)
    stopwords.remove('')
    return stopwords

# 下面这里有问题，和stopwords的比较无法正常进行
# 垃圾python2，编码问题，经由jieba处理的内容是unicode。但是stopwords里边的元素都是utf-8。在比对之前需要先将编码进行统一
def seg_sentence(sentence, stopwords):
    sentence_seged = pseg.cut(sentence.strip())
    outstr = []
    for seg in sentence_seged:
        word = seg.word.encode('utf-8')
        if word not in stopwords:
            if word != '\t':
                outstr.append(word)
    return ' '.join(outstr)

# 返回句子的生成器
def wordslist(data_file, stopwords):
    with open(data_file) as fp:
        line = fp.readline().strip()
        while line:
            parts = line.split('\t')
            # 判断当前行是否符合规范(id, date, title, content), 不符合则跳过，处理下一行
            if len(parts) != 4:
                print 'error line: ', line
                line = fp.readline().strip()
                continue
            id, date, title, content = parts
            content_seg = seg_sentence(content, stopwords)
            yield content_seg

def save_doc_date_kws(data_file, corpus, doc_keywords_file):
    with open(data_file) as fp, open(doc_keywords_file, 'w') as wfp:
        line = fp.readline().strip()
        line_counter = 0
        while line:
            parts = line.split('\t')
            # 判断当前行是否符合规范(id, date, title, content), 不符合则跳过，处理下一行(之前的废弃内容也没有被考虑进来）
            if len(parts) != 4:
                print 'error line: ', line
                line = fp.readline().strip()
                continue

            id = parts[0]
            date = parts[1]
            kws = corpus[line_counter]
            # 新加上去的约束，先把数字项给删了
            # 不行，不能通过删除remove进行操作
            kwsWithoutNum = []
            for kw in kws:
                if re.compile(r'-?[0-9]\d*').match(kw):
                    continue
                kwsWithoutNum.append(kw)

            date = date.split('T')[0]
            # 不满足如下条件不写入
            if len(kwsWithoutNum) > 4:
                wfp.write(id + ',' + date + ',' + '|'.join(kwsWithoutNum) + '\n')

            line = fp.readline().strip()
            line_counter += 1



if __name__ == '__main__':
    # 不行啊，拆分文件是为了能够更有效率地生成那个列表。毕竟后续的tfidf统计需要建立在全体语料库的基础上

    userdict = '/home/zhangchj/PycharmProjects/Analysis/config/userdict_merged4.txt'
    jieba.load_userdict(userdict)
    stopwords = stopwordslist('/home/zhangchj/PycharmProjects/Analysis/config/stopwords.txt')
    print 'finish loading stopwords'

    segSentencelist = list()
    for i in range(1, 5):
        segSentencelist += load_variavle('./data/segSentencelist_{}.variable'.format(i))

    print 'finish loading segSentencelist'

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(segSentencelist))
    words = vectorizer.get_feature_names()
    print 'finish calculating tfidf'

    save_variable(tfidf, './data/tfidf.variable')
    save_variable(words, './data/words.variable')


    corpus = []
    kwsNum = 10

    # 逐个把关键词添加到语料库中(应该加一个进度显示的)
    for i in range(tfidf.shape[0]):
        oneSegSentence_weights = tfidf[i].toarray()  # oneSegSentence_weights[0][loc[i]] 可以获取对应tfidf值按倒序顺位的值
        loc = np.argsort(-oneSegSentence_weights)[0] # 按倒序排列

        nonzero_num = tfidf.shape[1] - list(oneSegSentence_weights[0]).count(0)

        if nonzero_num < kwsNum: smaller = nonzero_num
        else: smaller = kwsNum
        kws = []
        for j in range(smaller):
            kws.append(words[loc[j]])

        corpus.append(kws)

        if i % 10000 == 0:
            print i

    save_variable(corpus, './data/corpus.variable')
    print 'finish saving corpus'

    doc_keywords_file = './res/topic_extraction/mine_2017_doc_kws.csv'

    data_file= './wengaoku/normalized/2017.data'
    save_doc_date_kws(data_file, corpus, doc_keywords_file)






