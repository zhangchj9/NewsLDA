# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
import pickle # 以防万一
import time
from multiprocessing import Pool

'''
从文章内容的结构化抽取（关键词提取）入手从头开始
这个list的过程太漫长了(3小时内无法解决)，得考虑先把数据文件进行拆分（用shell吧）
主要问题不是这个文件大，是这个yield，在处理这个文件的反复读写问题上大大拖慢了运行速度
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

# 创建停用词list(可以直接调用黑名单)
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  # 已经确保不含有重复内容
    stopwords = set(stopwords)
    stopwords.remove('')
    return stopwords


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
    content_seg = []
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
            content_seg.append(seg_sentence(content, stopwords))
            line = fp.readline().strip()
    return content_seg
    # yield content_seg # 这种yield不知道是否涉及反复读写文件，如果涉及到了那就是真的慢了


def run_proc(name, stopwords):
    start = time.time()
    print 'running process {}'.format(name)
    data_file = '/home/zhangchj/PycharmProjects/BLMYX01/wengaoku/normalized/split/2017_{}.data'.format(name)
    segSentencelist = wordslist(data_file, stopwords)
    end = time.time()
    save_variable(segSentencelist, './data/segSentencelist_{}.variable'.format(name))
    print 'finish loading segSentencelist_{}, task runs {:.2f} seconds'.format(name, end-start)


if __name__ == '__main__':
    # 不行啊，拆分文件是为了能够更有效率地生成那个列表。毕竟后续的tfidf统计需要建立在全体语料库的基础上

    userdict = '/home/zhangchj/PycharmProjects/Analysis/config/userdict_merged4.txt'
    jieba.load_userdict(userdict)
    stopwords = stopwordslist('/home/zhangchj/PycharmProjects/Analysis/config/stopwords.txt')
    print 'finish loading stopwords'

    p = Pool(4)
    for name in range(1,5):
        p.apply_async(run_proc, args=(name, stopwords, ))
    p.close()
    p.join()